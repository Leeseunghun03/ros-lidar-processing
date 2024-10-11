#include "../include/lidar_processing/lidar_processing_node.hpp"

using namespace std;

LidarProcessing::LidarProcessing()
{
    std::string sub_pcl_topic;
    ros::param::get("/lidar_processing_node/sub_pcl_topic", sub_pcl_topic);
    ROS_INFO("Subscribing PCL Topic : %s", sub_pcl_topic.c_str());

    std::string sub_state_topic;
    ros::param::get("/lidar_processing_node/sub_state_topic", sub_state_topic);
    ROS_INFO("Subscribing State Topic : %s", sub_state_topic.c_str());

    std::string pub_topic;
    ros::param::get("/lidar_processing_node/pub_topic", pub_topic);

    sub_pcl = nh.subscribe<sensor_msgs::PointCloud2>(sub_pcl_topic, 200000, &LidarProcessing::pcl_cbk, this);
    sub_state = nh.subscribe<std_msgs::String>(sub_state_topic, 1, &LidarProcessing::state_cbk, this);
    pub_pcl = nh.advertise<sensor_msgs::PointCloud2>(pub_topic, 100000);
}

void LidarProcessing::pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *cloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>());

    // Set Poincloud range
    set_roi_range(cloud, cloud_out);

    // Voxel filtering(Down Sampling)
    voxel_filtering(cloud_out);

    // Remove Outlier
    remove_outlier(cloud_out, 100, 0.1);

    // 평면 검출 및 분리
    pcl::PointCloud<pcl::PointXYZI>::Ptr inlierPoints(new pcl::PointCloud<pcl::PointXYZI>),
        inlierPoints_neg(new pcl::PointCloud<pcl::PointXYZI>);
    ransac_segmentation(cloud_out, inlierPoints, inlierPoints_neg);

    // Clustering(Object 군집화)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr result_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    DBSCAN_clustering(inlierPoints_neg, result_cloud);

    // Publishing
    pub_msg(msg, result_cloud);
}

void LidarProcessing::state_cbk(const std_msgs::String::ConstPtr &msg)
{
    state = msg->data;
    std::cout << "state: " << state << std::endl;
}

void LidarProcessing::pub_msg(const sensor_msgs::PointCloud2::ConstPtr &input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &result_cloud)
{
    pcl::toROSMsg(*result_cloud, result_msg);
    result_msg.header.frame_id = "map";
    result_msg.header.stamp = input->header.stamp;
    pub_pcl.publish(result_msg);
}

void LidarProcessing::remove_outlier(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, int meanK = 50, double stdDevMulThresh = 1.0)
{
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> outlier_removal;
    outlier_removal.setInputCloud(cloud);                // 입력 점군 설정
    outlier_removal.setMeanK(meanK);                     // 각 점에 대해 고려할 주변 이웃의 개수
    outlier_removal.setStddevMulThresh(stdDevMulThresh); // 표준 편차 임계값
    outlier_removal.filter(*cloud);
}

void LidarProcessing::DBSCAN_clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &result_cloud)
{
    arma::mat data(3, input_cloud->points.size());

    for (size_t i = 0; i < input_cloud->points.size(); ++i)
    {
        data(0, i) = input_cloud->points[i].x;
        data(1, i) = input_cloud->points[i].y;
        data(2, i) = input_cloud->points[i].z;
    }

    mlpack::dbscan::DBSCAN<> dbscan(epsilon, minPoints);

    arma::Row<size_t> assignments;
    dbscan.Cluster(data, assignments);

    for (size_t i = 0; i < assignments.n_elem; ++i)
    {
        if (input_cloud->points[i].z < -0.2)
        {
            continue;
        }
        pcl::PointXYZRGB point;
        point.x = input_cloud->points[i].x;
        point.y = input_cloud->points[i].y;
        point.z = input_cloud->points[i].z;

        size_t color_index = assignments[i] % COLORS.size();
        point.r = COLORS[color_index][0];
        point.g = COLORS[color_index][1];
        point.b = COLORS[color_index][2];

        result_cloud->points.push_back(point);
    }
}

void LidarProcessing::ransac_segmentation(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out, pcl::PointCloud<pcl::PointXYZI>::Ptr &inlierPoints, pcl::PointCloud<pcl::PointXYZI>::Ptr &inlierPoints_neg)
{
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE); // 적용 모델
    seg.setMethodType(pcl::SAC_RANSAC);    // 적용 방법
    seg.setMaxIterations(2000);            // 최대 실행 수
    seg.setDistanceThreshold(0.08);        // inlier로 처리할 거리 정보
    seg.setInputCloud(cloud_out);          // 입력
    seg.segment(*inliers, *coefficients);  // 세그멘테이션 적용

    pcl::copyPointCloud<pcl::PointXYZI>(*cloud_out, *inliers, *inlierPoints);

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud_out);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*inlierPoints_neg);
}

void LidarProcessing::voxel_filtering(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
{
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    pcl::PCLPointCloud2::Ptr cloud_voxel(new pcl::PCLPointCloud2);
    pcl::PCLPointCloud2::Ptr cloud_PCL2(new pcl::PCLPointCloud2);
    pcl::toPCLPointCloud2(*cloud, *cloud_PCL2);
    sor.setInputCloud(cloud_PCL2);

    if (state == "inside")
        sor.setLeafSize(0.2f, 0.2f, 0.2f);
    else if (state == "outside")
        sor.setLeafSize(0.4f, 0.4f, 0.4f);
    else
        sor.setLeafSize(0.1f, 0.1f, 0.1f);

    sor.filter(*cloud_voxel);
    pcl::fromPCLPointCloud2(*cloud_voxel, *cloud);
}

void LidarProcessing::set_roi_range(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out)
{
    pcl::PassThrough<pcl::PointXYZI> pass;

    if (state == "inside")
    {
        pass.setInputCloud(cloud); // raw data 입력
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-2, 2); // 상하거리
        pass.filter(*cloud_out);

        pass.setFilterFieldName("x");
        pass.setFilterLimits(0, 60); // 앞뒤거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-8.0, 8.0); // 좌우거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);
    }
    else if (state == "outside")
    {
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-2, 2); // 상하거리
        pass.filter(*cloud_out);

        pass.setFilterFieldName("x");
        pass.setFilterLimits(0, 60); // 앞뒤거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-8.0, 8.0); // 좌우거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);
    }
    else
    {
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-5, 13); // 상하거리
        pass.filter(*cloud_out);

        pass.setFilterFieldName("x");
        pass.setFilterLimits(0, 20); // 앞뒤거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);

        pass.setFilterFieldName("y");
        pass.setFilterLimits(-10, 10); // 좌우거리
        pass.setInputCloud(cloud_out);
        pass.filter(*cloud_out);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_processing_node");
    ROS_INFO("Starting Lidar Processing Node");

    LidarProcessing lp;

    ros::Rate loop_rate(100);
    while (ros::ok())
    {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}