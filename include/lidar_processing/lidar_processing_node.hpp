#ifndef LIDAR_PROCESSING_HPP
#define LIDAR_PROCESSING_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <std_msgs/String.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <mlpack/core.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>
#include "color.hpp"

class LidarProcessing
{
public:
    LidarProcessing();

private:
    void pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
    void state_cbk(const std_msgs::String::ConstPtr &msg);
    void set_roi_range(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out);
    void voxel_filtering(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);
    void ransac_segmentation(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out, pcl::PointCloud<pcl::PointXYZI>::Ptr &inlierPoints, pcl::PointCloud<pcl::PointXYZI>::Ptr &inlierPoints_neg);
    void DBSCAN_clustering(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &result_cloud);
    void remove_outlier(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, int meanK, double stdDevMulThresh);
    void pub_msg(const sensor_msgs::PointCloud2::ConstPtr& input, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& result_cloud);

    sensor_msgs::PointCloud2 result_msg;

    ros::NodeHandle nh;
    ros::Subscriber sub_pcl;
    ros::Subscriber sub_state;
    ros::Publisher pub_pcl;
    std::string state;

    double epsilon = 0.3;
    size_t minPoints = 3;
};

#endif // LIDAR_PROCESSING_HPP
