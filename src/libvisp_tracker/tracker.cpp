#include <stdexcept>

#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <boost/scope_exit.hpp>
#include <boost/version.hpp>

#include <dynamic_reconfigure/server.h>
#include <image_proc/advertisement_checker.h>
#include <image_transport/image_transport.h>
#include <ros/param.h>
#include <ros/ros.h>
#include <ros/transport_hints.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>
#include <tf/transform_broadcaster.h>

#include <visp_tracker/MovingEdgeSites.h>

#include <boost/bind.hpp>
#include <visp/vpExponentialMap.h>
#include <visp/vpImage.h>
#include <visp/vpImageConvert.h>
#include <visp/vpCameraParameters.h>
#include <visp/vpMbEdgeTracker.h>

#include "conversion.hh"
#include "callbacks.hh"
#include "file.hh"
#include "names.hh"

#include "tracker.hh"

// TODO:
// - add a topic allowing to suggest an estimation of the cMo
// - handle automatic reset when tracking is lost.

namespace visp_tracker
{
  bool
  Tracker::initCallback(visp_tracker::Init::Request& req,
			visp_tracker::Init::Response& res)
  {
    ROS_INFO("Initialization request received.");

    res.initialization_succeed = false;

    // If something goes wrong, rollback all changes.
    BOOST_SCOPE_EXIT((&res)(&tracker_)(&state_)
		     (&lastTrackedImage_))
      {
	if(!res.initialization_succeed)
	  {
	    tracker_.resetTracker();
	    state_ = WAITING_FOR_INITIALIZATION;
	    lastTrackedImage_ = 0;
	  }
      } BOOST_SCOPE_EXIT_END;

    std::string fullModelPath;
    boost::filesystem::ofstream modelStream;

    // Load model from parameter.
    if (!makeModelFile(modelStream, fullModelPath))
      return true;

    // Load moving edges.
    vpMe movingEdge;
    convertInitRequestToVpMe(req, tracker_, movingEdge);

    // Update parameters.
    visp_tracker::MovingEdgeConfig config;
    convertVpMeToMovingEdgeConfig(movingEdge, tracker_, config);
    reconfigureSrv_.updateConfig(config);

    //FIXME: not sure if this is needed.
    movingEdge.initMask();

    // Reset the tracker and the node state.
    tracker_.resetTracker();
    state_ = WAITING_FOR_INITIALIZATION;
    lastTrackedImage_ = 0;

    tracker_.setMovingEdge(movingEdge);

    // Load the model.
    try
      {
	ROS_DEBUG_STREAM("Trying to load the model: " << fullModelPath);
	tracker_.loadModel(fullModelPath.c_str());
	modelStream.close();
      }
    catch(...)
      {
	ROS_ERROR_STREAM("Failed to load the model: " << fullModelPath);
	return true;
      }
    ROS_DEBUG("Model has been successfully loaded.");

    // Load the initial cMo.
    transformToVpHomogeneousMatrix(cMo_, req.initial_cMo);

    // Try to initialize the tracker.
    ROS_INFO_STREAM("Initializing tracker with cMo:\n" << cMo_);
    try
      {
	tracker_.init(image_, cMo_);
	ROS_INFO("Tracker successfully initialized.");

	movingEdge.print();
      }
    catch(const std::string& str)
      {
	ROS_ERROR_STREAM("Tracker initialization has failed: " << str);
      }

    // Initialization is valid.
    res.initialization_succeed = true;
    state_ = TRACKING;
    return true;
  }

  void
  Tracker::updateMovingEdgeSites(visp_tracker::MovingEdgeSitesPtr sites)
  {
    if (!sites)
      return;

    std::list<vpMbtDistanceLine*> linesList;
    tracker_.getLline(linesList, 0);

    std::list<vpMbtDistanceLine*>::iterator linesIterator =
      linesList.begin();
    if (linesList.empty())
      ROS_DEBUG_THROTTLE(10, "no distance lines");
    bool noVisibleLine = true;
    for (; linesIterator != linesList.end(); ++linesIterator)
      {
	vpMbtDistanceLine* line = *linesIterator;

	if (line && line->isVisible())
	  {
	    std::list<vpMeSite>::const_iterator sitesIterator =
	      line->meline->list.begin();
	    if (line->meline->list.empty())
	      ROS_DEBUG_THROTTLE(10, "no moving edge for a line");
	    for (; sitesIterator != line->meline->list.end(); ++sitesIterator)
	      {
		visp_tracker::MovingEdgeSite movingEdgeSite;
		movingEdgeSite.x = sitesIterator->ifloat;
		movingEdgeSite.y = sitesIterator->jfloat;
		movingEdgeSite.suppress = sitesIterator->suppress;
		sites->moving_edge_sites.push_back (movingEdgeSite);
	      }
	    noVisibleLine = false;
	  }
      }
    if (noVisibleLine)
      ROS_DEBUG_THROTTLE(10, "no distance lines");
  }

  void Tracker::checkInputs()
  {
    ros::V_string topics;
    topics.push_back(rectifiedImageTopic_);
    checkInputs_.start(topics, 60.0);
  }

  Tracker::Tracker(ros::NodeHandle& nh,
		   ros::NodeHandle& privateNh,
		   volatile bool& exiting,
		   unsigned queueSize)
    : exiting_ (exiting),
      queueSize_(queueSize),
      nodeHandle_(nh),
      nodeHandlePrivate_(privateNh),
      imageTransport_(nodeHandle_),
      state_(WAITING_FOR_INITIALIZATION),
      image_(),
      cameraPrefix_(),
      rectifiedImageTopic_(),
      cameraInfoTopic_(),
      vrmlPath_(),
      cameraSubscriber_(),
      mutex_ (),
      reconfigureSrv_(mutex_, nodeHandlePrivate_),
      resultPublisher_(),
      transformationPublisher_(),
      movingEdgeSitesPublisher_(),
      initService_(),
      header_(),
      info_(),
      movingEdge_(),
      cameraParameters_(),
      tracker_(),
      lastTrackedImage_(),
      checkInputs_(nodeHandle_, ros::this_node::getName()),
      cMo_ (),
      listener_ (),
      worldFrameId_ (),
      compensateRobotMotion_ (false),
      transformBroadcaster_ (),
      childFrameId_ ()
  {
    // Set cMo to identity.
    cMo_.eye();

    // Parameters.
    nodeHandlePrivate_.param<std::string>("camera_prefix", cameraPrefix_, "");

    if (cameraPrefix_.empty ())
      {
	ROS_FATAL
	  ("The camera_prefix parameter not set.\n"
	   "Please relaunch the tracker while setting this parameter, i.e.\n"
	   "$ rosrun visp_tracker tracker _camera_prefix:=/my/camera");
	ros::shutdown ();
	return;
      }
    nodeHandle_.setParam("camera_prefix", cameraPrefix_);

    nodeHandle_.param<std::string>("frame_id", childFrameId_, "object_position");

    // Robot motion compensation.
    nodeHandle_.param<std::string>("world_frame_id", worldFrameId_, "/odom");
    nodeHandle_.param<bool>
      ("compensate_robot_motion", compensateRobotMotion_, false);

    // Compute topic and services names.
    rectifiedImageTopic_ =
      ros::names::resolve(cameraPrefix_ + "/image_rect");

    // Check for subscribed topics.
    checkInputs();

    // Result publisher.
    resultPublisher_ =
      nodeHandle_.advertise<visp_tracker::TrackingResult>
      (visp_tracker::result_topic, queueSize_);

    transformationPublisher_ =
      nodeHandle_.advertise<geometry_msgs::TransformStamped>
      (visp_tracker::object_position_topic, queueSize_);

    // Moving edge sites_ publisher.
    movingEdgeSitesPublisher_ =
      nodeHandle_.advertise<visp_tracker::MovingEdgeSites>
      (visp_tracker::moving_edge_sites_topic, queueSize_);

    // Camera subscriber.
    cameraSubscriber_ =
      imageTransport_.subscribeCamera
      (rectifiedImageTopic_, queueSize_,
       bindImageCallback(image_, header_, info_));

    // Initialization.
    movingEdge_.initMask();
    tracker_.setMovingEdge(movingEdge_);

    // Dynamic reconfigure.
    reconfigureSrv_t::CallbackType f =
      boost::bind(&reconfigureCallback, boost::ref(tracker_),
		  boost::ref(image_), boost::ref(movingEdge_),
		  boost::ref(mutex_), _1, _2);
    reconfigureSrv_.setCallback(f);

    // Wait for the image to be initialized.
    waitForImage();
    if (this->exiting())
      return;
    if (!image_.getWidth() || !image_.getHeight())
      throw std::runtime_error("failed to retrieve image");

    // Tracker initialization.
    initializeVpCameraFromCameraInfo(cameraParameters_, info_);

    // Double check camera parameters.

    if (cameraParameters_.get_px () == 0.
	|| cameraParameters_.get_px () == 1.
	|| cameraParameters_.get_py () == 0.
	|| cameraParameters_.get_py () == 1.
	|| cameraParameters_.get_u0 () == 0.
	|| cameraParameters_.get_u0 () == 1.
	|| cameraParameters_.get_v0 () == 0.
	|| cameraParameters_.get_v0 () == 1.)
      ROS_WARN ("Dubious camera parameters detected.\n"
		"\n"
		"It seems that the matrix P from your camera\n"
		"calibration topics is wrong.\n"
		"The tracker will continue anyway, but you\n"
		"should double check your calibration data,\n"
		"especially if the model re-projection fails.\n"
		"\n"
		"This warning is triggered is px, py, u0 or v0\n"
		"is set to 0. or 1. (exactly).");

    tracker_.setCameraParameters(cameraParameters_);
    tracker_.setDisplayMovingEdges(false);

    ROS_INFO_STREAM(cameraParameters_);

    // Service declaration.
    initCallback_t initCallback =
      boost::bind(&Tracker::initCallback, this, _1, _2);

    initService_ = nodeHandle_.advertiseService
      (visp_tracker::init_service, initCallback);
  }

  void Tracker::spin()
  {
    ros::Rate loopRateTracking(100);
    tf::Transform transform;
    std_msgs::Header lastHeader;

    while (!exiting())
      {
	// When a camera sequence is played several times,
	// the seq id will decrease, in this case we want to
	// continue the tracking.
	if (header_.seq < lastHeader.seq)
	  lastTrackedImage_ = 0;

	if (lastTrackedImage_ < header_.seq)
	  {
	    lastTrackedImage_ = header_.seq;

	    // If we can estimate the camera displacement using tf,
	    // we update the cMo to compensate for robot motion.
	    if (compensateRobotMotion_)
	      try
		{
		  tf::StampedTransform stampedTransform;
		  listener_.lookupTransform
		    (header_.frame_id, // camera frame name
		     header_.stamp,    // current image time
		     header_.frame_id, // camera frame name
		     lastHeader.stamp, // last processed image time
		     worldFrameId_,    // frame attached to the environment
		     stampedTransform
		     );
		  vpHomogeneousMatrix newMold;
		  transformToVpHomogeneousMatrix (newMold, stampedTransform);
		  cMo_ = newMold * cMo_;
		}
	      catch(tf::TransformException& e)
		{}

	    if (state_ == TRACKING || state_ == LOST)
	      try
		{
		  tracker_.init(image_, cMo_);
		  tracker_.track(image_);
		  tracker_.getPose(cMo_);
		}
	      catch(...)
		{
		  ROS_WARN_THROTTLE(10, "tracking lost");
		  state_ = LOST;
		}

	    // Publish the tracking result.
	    if (state_ == TRACKING)
	      {
		geometry_msgs::Transform transformMsg;
		vpHomogeneousMatrixToTransform(transformMsg, cMo_);

		// Publish position.
		if (transformationPublisher_.getNumSubscribers	() > 0)
		  {
		    geometry_msgs::TransformStampedPtr objectPosition
		      (new geometry_msgs::TransformStamped);
		    objectPosition->header = header_;
		    objectPosition->child_frame_id = childFrameId_;
		    objectPosition->transform = transformMsg;
		    transformationPublisher_.publish(objectPosition);
		  }

		// Publish result.
		if (resultPublisher_.getNumSubscribers	() > 0)
		  {
		    visp_tracker::TrackingResultPtr result
		      (new visp_tracker::TrackingResult);
		    result->header = header_;
		    result->is_tracking = true;
		    result->cMo = transformMsg;
		    resultPublisher_.publish(result);
		  }

		// Publish moving edge sites.
		if (movingEdgeSitesPublisher_.getNumSubscribers	() > 0)
		  {
		    visp_tracker::MovingEdgeSitesPtr sites
		      (new visp_tracker::MovingEdgeSites);
		    updateMovingEdgeSites(sites);
		    sites->header = header_;
		    movingEdgeSitesPublisher_.publish(sites);
		  }

		// Publish to tf.
		transform.setOrigin
		  (tf::Vector3(transformMsg.translation.x,
			       transformMsg.translation.y,
			       transformMsg.translation.z));
		transform.setRotation
		  (tf::Quaternion
		   (transformMsg.rotation.x,
		    transformMsg.rotation.y,
		    transformMsg.rotation.z,
		    transformMsg.rotation.w));
		transformBroadcaster_.sendTransform
		  (tf::StampedTransform
		   (transform,
		    header_.stamp,
		    header_.frame_id,
		    childFrameId_));
	      }
	    else if (resultPublisher_.getNumSubscribers	() > 0)
	      {
		visp_tracker::TrackingResultPtr result
		  (new visp_tracker::TrackingResult);
		result->header = header_;
		result->is_tracking = false;
		result->cMo.translation.x = 0.;
		result->cMo.translation.y = 0.;
		result->cMo.translation.z = 0.;

		result->cMo.rotation.x = 0.;
		result->cMo.rotation.y = 0.;
		result->cMo.rotation.z = 0.;
		result->cMo.rotation.w = 0.;
		resultPublisher_.publish(result);
	      }
	  }
	lastHeader = header_;

	spinOnce();
	loopRateTracking.sleep();
      }
  }

  // Make sure that we have an image *and* associated calibration
  // data.
  void
  Tracker::waitForImage()
  {
    ros::Rate loop_rate(10);
    while (!exiting()
	   && (!image_.getWidth() || !image_.getHeight())
	   && (!info_ || info_->K[0] == 0.))
      {
	ROS_INFO_THROTTLE(1, "waiting for a rectified image...");
	spinOnce();
	loop_rate.sleep();
      }
  }

} // end of namespace visp_tracker.
