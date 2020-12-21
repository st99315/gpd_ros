#include <gpd_ros/grasp_detection_node.h>


/** constants for input point cloud types */
const int GraspDetectionNode::POINT_CLOUD_2 = 0; ///< sensor_msgs/PointCloud2
const int GraspDetectionNode::CLOUD_INDEXED = 1; ///< cloud with indices
const int GraspDetectionNode::CLOUD_SAMPLES = 2; ///< cloud with (x,y,z) samples


GraspDetectionNode::GraspDetectionNode(ros::NodeHandle& node) : has_cloud_(false), has_normals_(false),
  size_left_cloud_(0), has_samples_(true), frame_(""), use_importance_sampling_(false), goal_active_(false)
{
  printf("Init ....\n");

  // set camera viewpoint to default origin
  // std::vector<double> camera_position;
  // node.getParam("camera_position", camera_position);
  // view_point_ << camera_position[0], camera_position[1], camera_position[2];
  view_point_ << 0, 0, 0;

  // choose sampling method for grasp detection
//  node.param("use_importance_sampling", use_importance_sampling_, false);

//  if (use_importance_sampling_)
//  {
//    importance_sampling_ = new SequentialImportanceSampling(node);
//  }
  std::string cfg_file;
  node.param("config_file", cfg_file, std::string(""));
  grasp_detector_.reset(new gpd::GraspDetector(cfg_file));
  printf("Created GPD ....\n");

  // Read input cloud and sample ROS topics parameters.
  int cloud_type;
  node.param("cloud_type", cloud_type, POINT_CLOUD_2);
  std::string cloud_topic;
  node.param("cloud_topic", cloud_topic, std::string("/camera/depth/color/points"));
  std::string samples_topic;
  node.param("samples_topic", samples_topic, std::string(""));
  std::string rviz_topic;
  node.param("rviz_topic", rviz_topic, std::string("plot_grasps"));

  if (!rviz_topic.empty()) {
    rviz_plotter_.reset(new GraspPlotter(node, grasp_detector_->getHandSearchParameters().hand_geometry_));
  }

  // subscribe to input point cloud ROS topic
  if (cloud_type == POINT_CLOUD_2) {
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloudCallback, this);
  } else if (cloud_type == CLOUD_INDEXED) {
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloudIndexedCallback, this);
  } else if (cloud_type == CLOUD_SAMPLES) {
    cloud_sub_ = node.subscribe(cloud_topic, 1, &GraspDetectionNode::cloudSamplesCallback, this);
    //    grasp_detector_->setUseIncomingSamples(true);
    has_samples_ = false;
  }

  // subscribe to input samples ROS topic
  if (!samples_topic.empty()) {
    samples_sub_ = node.subscribe(samples_topic, 1, &GraspDetectionNode::samplesCallback, this);
    has_samples_ = false;
  }

  node.getParam("workspace", workspace_);

  node.param("robot_base", robot_frame_, std::string(""));
  node.param("camera_optical", cam_opt_frame_, std::string(""));

  node.param("filtering_score", filtering_score_, true);
  node.param("score_threshold", score_threshold_, 0.);

  std::string action_name;
  node.param("action_name", action_name, std::string(""));

  if (action_name.empty()) {
    service_driven_ = false;
    // uses ROS topics to publish grasp candidates, antipodal grasps, and grasps after clustering
    grasps_pub_ = node.advertise<gpd_ros::GraspConfigList>("clustered_grasps", 10);
  }
  else {
    service_driven_ = true;
    server_.reset(new actionlib::SimpleActionServer<gpd_ros::SampleGraspPosesAction>(
        node, action_name, false));
    server_->registerGoalCallback(std::bind(&GraspDetectionNode::goalCallback, this));
    server_->registerPreemptCallback(std::bind(&GraspDetectionNode::preemptCallback, this));
    server_->start();
  }
}


void GraspDetectionNode::run()
{
  ros::Rate rate(100);
  ROS_INFO("Waiting for point cloud to arrive ...");

  while (ros::ok()) {
    if (has_cloud_) {
      // Detect grasps in point cloud.
      auto grasps = detectGraspPoses();
      if (service_driven_) {
        filterByScore(grasps);
        sampleGrasps(grasps);
      }

      // Visualize the detected grasps in rviz.
      if (rviz_plotter_) {
        rviz_plotter_->drawGrasps(grasps, frame_);
      }

      // Reset the system.
      has_cloud_ = false;
      has_samples_ = false;
      has_normals_ = false;
      goal_active_ = false;
      ROS_INFO("Waiting for point cloud to arrive ...");
    }

    ros::spinOnce();
    rate.sleep();
  }
}


bool GraspDetectionNode::getEigenTransform(const std::string& src, const std::string& dst, Eigen::Isometry3d& out)
{
  static tf2_ros::Buffer tfBuffer;
  static tf2_ros::TransformListener tfListener(tfBuffer);

  try {
    auto transform_stamped = tfBuffer.lookupTransform(
        src, dst,
        ros::Time(0),
        ros::Duration(1.));
    out = tf2::transformToEigen(transform_stamped);
  }
  catch (tf2::TransformException &ex) {
    ROS_WARN("%s",ex.what());
    return false;
  }
  return true;
}


void GraspDetectionNode::filterByScore(std::vector<std::unique_ptr<gpd::candidate::Hand>>& grasps)
{
  if (!filtering_score_)  return;
  ROS_INFO("Filtering grasps by score, original size: %lu", grasps.size());

  size_t i = 0;
  while (i < grasps.size()) {
    if (grasps.at(i)->getScore() < score_threshold_) {
      grasps.erase(grasps.begin() + i);
    }
    else {
      ++i;
    }
  }
  ROS_INFO("Filtered size: %lu", grasps.size());
}


void GraspDetectionNode::sampleGrasps(const std::vector<std::unique_ptr<gpd::candidate::Hand>>& grasps)
{
  gpd_ros::SampleGraspPosesResult result;

  if (grasps.empty()) {
    ROS_ERROR("No grasp candidates found with a positive cost");
    result.grasp_state = "failed";
    server_->setAborted(result);
    return;
  }

  // TODO: both eyeinhand and eyetohand
  Eigen::Isometry3d trans_base_cam_opt;
  if (!getEigenTransform(base_frame_, cam_opt_frame_, trans_base_cam_opt)) {
    ROS_ERROR("Transform is not found from %s to %s", base_frame_.c_str(), cam_opt_frame_.c_str());
    result.grasp_state = "failed";
    server_->setAborted(result);
    return;
  }

  for (const auto& hand: grasps) {
    // transform grasp from camera optical link into frame_id
    const Eigen::Isometry3d transform_opt_grasp =
        Eigen::Translation3d(hand->getPosition()) * Eigen::Quaterniond(hand->getOrientation());

    const Eigen::Isometry3d cvt_rot = cvt_poses_?
        getTransformFromRPY(-1.57, 0, -1.57): Eigen::Isometry3d::Identity();
    const Eigen::Isometry3d transform_base_grasp = trans_base_cam_opt * transform_opt_grasp * cvt_rot;
    const Eigen::Vector3d trans = transform_base_grasp.translation();
    const Eigen::Quaterniond rot(transform_base_grasp.rotation());

    // convert back to PoseStamped
    geometry_msgs::PoseStamped grasp_pose;
    grasp_pose.header.frame_id = base_frame_;
    grasp_pose.pose.position.x = trans.x();
    grasp_pose.pose.position.y = trans.y();
    grasp_pose.pose.position.z = trans.z();

    grasp_pose.pose.orientation.w = rot.w();
    grasp_pose.pose.orientation.x = rot.x();
    grasp_pose.pose.orientation.y = rot.y();
    grasp_pose.pose.orientation.z = rot.z();

    result.grasp_candidates.emplace_back(grasp_pose);

    // Grasp is selected based on cost not score
    // Invert score to represent grasp with lowest cost
    result.costs.emplace_back(static_cast<double>(1.0 / hand->getScore()));
  }

  result.grasp_state = "success";
  server_->setSucceeded(result);
}


Eigen::Isometry3d GraspDetectionNode::getTransformFromRPY(double rx, double ry, double rz)
{
  Eigen::AngleAxisd x_angle(rx, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd y_angle(ry, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd z_angle(rz, Eigen::Vector3d::UnitZ());

  Eigen::Quaterniond q = z_angle * y_angle * x_angle;
  return (Eigen::Translation3d(0, 0, 0) * q);
}


std::vector<std::unique_ptr<gpd::candidate::Hand>> GraspDetectionNode::detectGraspPoses()
{
  // detect grasp poses
  std::vector<std::unique_ptr<gpd::candidate::Hand>> grasps;

  if (use_importance_sampling_)
  {
//    cloud_camera_->filterWorkspace(workspace_);
//    cloud_camera_->voxelizeCloud(0.003);
//    cloud_camera_->calculateNormals(4);
//    grasps = importance_sampling_->detectGrasps(*cloud_camera_);
    printf("Error: importance sampling is not supported yet\n");
  }
  else
  {
    // preprocess the point cloud
    grasp_detector_->preprocessPointCloud(*cloud_camera_);

    // detect grasps in the point cloud
    grasps = grasp_detector_->detectGrasps(*cloud_camera_);
  }

  if (!service_driven_) {
    // Publish the selected grasps.
    gpd_ros::GraspConfigList selected_grasps_msg = GraspMessages::createGraspListMsg(grasps, cloud_camera_header_);
    grasps_pub_.publish(selected_grasps_msg);
    ROS_INFO_STREAM("Published " << selected_grasps_msg.grasps.size() << " highest-scoring grasps.");
  }
  return grasps;
}


std::vector<int> GraspDetectionNode::getSamplesInBall(const PointCloudRGBA::Ptr& cloud,
  const pcl::PointXYZRGBA& centroid, float radius)
{
  std::vector<int> indices;
  std::vector<float> dists;
  pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
  kdtree.setInputCloud(cloud);
  kdtree.radiusSearch(centroid, radius, indices, dists);
  return indices;
}


void GraspDetectionNode::goalCallback()
{
  const auto& goal = server_->acceptNewGoal();
  base_frame_ = goal->frame_id;
  base_frame_ = base_frame_.empty()? robot_frame_: base_frame_;
  cvt_poses_ = goal->cvt_poses;
  ROS_INFO("New goal accepted: using %s frame", base_frame_.c_str());
  goal_active_ = true;
}


void GraspDetectionNode::preemptCallback()
{
  ROS_INFO("Preempted gpd_ros");
  server_->setPreempted();
}


void GraspDetectionNode::cloudCallback(const sensor_msgs::PointCloud2& msg)
{
  if ((!has_cloud_ && !service_driven_)
      || (!has_cloud_ && service_driven_ && goal_active_))
  {
    Eigen::Matrix3Xd view_points(3,1);
    view_points.col(0) = view_point_;

    if (msg.fields.size() == 6 && msg.fields[3].name == "normal_x" && msg.fields[4].name == "normal_y"
      && msg.fields[5].name == "normal_z")
    {
      PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
      pcl::fromROSMsg(msg, *cloud);
      cloud_camera_.reset(new gpd::util::Cloud(cloud, 0, view_points));
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points and normals.");
    }
    else
    {
      PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
      pcl::fromROSMsg(msg, *cloud);

      // remove table plane
      if (true) {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.025);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);
        for (size_t i = 0; i < inliers->indices.size(); ++i) {
          cloud->points[inliers->indices[i]].x = std::numeric_limits<float>::quiet_NaN();
          cloud->points[inliers->indices[i]].y = std::numeric_limits<float>::quiet_NaN();
          cloud->points[inliers->indices[i]].z = std::numeric_limits<float>::quiet_NaN();
        }
      }

      cloud_camera_.reset(new gpd::util::Cloud(cloud, 0, view_points));
      cloud_camera_header_ = msg.header;
      ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points.");
    }

    frame_ = msg.header.frame_id;
    has_cloud_ = true;
  }
}


void GraspDetectionNode::cloudIndexedCallback(const gpd_ros::CloudIndexed& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the indices at which to sample grasp candidates.
    std::vector<int> indices(msg.indices.size());
    for (int i=0; i < indices.size(); i++)
    {
      indices[i] = msg.indices[i].data;
    }
    cloud_camera_->setSampleIndices(indices);

    has_cloud_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << msg.indices.size() << " samples");
  }
}


void GraspDetectionNode::cloudSamplesCallback(const gpd_ros::CloudSamples& msg)
{
  if (!has_cloud_)
  {
    initCloudCamera(msg.cloud_sources);

    // Set the samples at which to sample grasp candidates.
    Eigen::Matrix3Xd samples(3, msg.samples.size());
    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }
    cloud_camera_->setSamples(samples);

    has_cloud_ = true;
    has_samples_ = true;
    frame_ = msg.cloud_sources.cloud.header.frame_id;

    ROS_INFO_STREAM("Received cloud with " << cloud_camera_->getCloudProcessed()->size() << " points, and "
      << cloud_camera_->getSamples().cols() << " samples");
  }
}


void GraspDetectionNode::samplesCallback(const gpd_ros::SamplesMsg& msg)
{
  if (!has_samples_)
  {
    Eigen::Matrix3Xd samples(3, msg.samples.size());

    for (int i=0; i < msg.samples.size(); i++)
    {
      samples.col(i) << msg.samples[i].x, msg.samples[i].y, msg.samples[i].z;
    }

    cloud_camera_->setSamples(samples);
    has_samples_ = true;

    ROS_INFO_STREAM("Received grasp samples message with " << msg.samples.size() << " samples");
  }
}


void GraspDetectionNode::initCloudCamera(const gpd_ros::CloudSources& msg)
{
  // Set view points.
  Eigen::Matrix3Xd view_points(3, msg.view_points.size());
  for (int i = 0; i < msg.view_points.size(); i++)
  {
    view_points.col(i) << msg.view_points[i].x, msg.view_points[i].y, msg.view_points[i].z;
  }

  // Set point cloud.
  if (msg.cloud.fields.size() == 6 && msg.cloud.fields[3].name == "normal_x"
    && msg.cloud.fields[4].name == "normal_y" && msg.cloud.fields[5].name == "normal_z")
  {
    PointCloudPointNormal::Ptr cloud(new PointCloudPointNormal);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_.reset(new gpd::util::Cloud(cloud, camera_source, view_points));
  }
  else
  {
    PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
    pcl::fromROSMsg(msg.cloud, *cloud);

    // TODO: multiple cameras can see the same point
    Eigen::MatrixXi camera_source = Eigen::MatrixXi::Zero(view_points.cols(), cloud->size());
    for (int i = 0; i < msg.camera_source.size(); i++)
    {
      camera_source(msg.camera_source[i].data, i) = 1;
    }

    cloud_camera_.reset(new gpd::util::Cloud(cloud, camera_source, view_points));
    std::cout << "view_points:\n" << view_points << "\n";
  }
}


int main(int argc, char** argv)
{
  // seed the random number generator
  std::srand(std::time(0));

  // initialize ROS
  ros::init(argc, argv, "detect_grasps");
  ros::NodeHandle node("~");

  GraspDetectionNode grasp_detection(node);
  grasp_detection.run();

  return 0;
}
