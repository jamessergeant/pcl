/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id:$
 * @author: Koen Buys
 */

#ifndef PCL_GPU_SEGMENTATION_IMPL_EXTRACT_CLUSTERS_H_
#define PCL_GPU_SEGMENTATION_IMPL_EXTRACT_CLUSTERS_H_

#include <pcl/gpu/segmentation/gpu_extract_clusters.h>
#include <pcl/common/copy_point.h>

template <typename PointT> void
pcl::gpu::extractEuclideanClusters (const boost::shared_ptr<pcl::PointCloud<PointT> >  &host_cloud_,
                                    const pcl::gpu::Octree::Ptr                               &tree,
                                    float                                                     tolerance,
                                    std::vector<PointIndices>                                 &clusters,
                                    unsigned int                                              min_pts_per_cluster,
                                    unsigned int                                              max_pts_per_cluster,
                                    int                                                       min_query_points)
{

  // Create a bool vector of processed point indices, and initialize it to false
  // cloud is a DeviceArray<PointType>
  std::vector<bool> processed (host_cloud_->points.size (), false);

  int max_answers;

  if(max_pts_per_cluster > host_cloud_->points.size())
    max_answers = host_cloud_->points.size();
  else
    max_answers = max_pts_per_cluster;

  // to store the current cluster
  pcl::PointIndices r;

  DeviceArray<pcl::PointXYZ> queries_device_buffer;
  queries_device_buffer.create(max_answers);

  PointT t;
  pcl::PointXYZ point;

  // Create the query queue on the device, point based not indices
  pcl::gpu::Octree::Queries queries_device;
  // Create the query queue on the host
  pcl::PointCloud<pcl::PointXYZ>::VectorType queries_host;

  // Host buffer for results
  std::vector<int> sizes, data;

  // Process all points in the cloud
  for (size_t i = 0; i < host_cloud_->points.size (); ++i)
  {
    // if we already processed this point continue with the next one
    if (processed[i])
      continue;
    // now we will process this point
    processed[i] = true;

    // 
    copyPoint(host_cloud_->points[i], point);
    
    // Push the starting point in the vector
    queries_host.clear();
    queries_host.push_back (point);
    // Clear vector
    r.indices.clear();
    // Push the starting point in
    r.indices.push_back(static_cast<int> (i));

    unsigned int found_points = static_cast<unsigned int> (queries_host.size ());
    unsigned int previous_found_points = 0;

    pcl::gpu::NeighborIndices result_device;
    queries_device = DeviceArray<pcl::PointXYZ>(queries_device_buffer.ptr(),queries_host.size());

    // once the area stop growing, stop also iterating.
    do
    {
      sizes.clear();
      data.clear();
      // if the number of queries is not high enough implement search on Host here
      if(queries_host.size () <= min_query_points) ///@todo: adjust this to a variable number settable with method
      {
        for(size_t p = 0; p < queries_host.size (); p++)
        {
          // Execute the radiusSearch on the host
          tree->radiusSearchHost(queries_host[p], tolerance, data, max_answers);
        }
        // Store the previously found number of points
        previous_found_points = found_points;
        // Clear queries list
        queries_host.clear();

        //std::unique(data.begin(), data.end());
        if(data.size () == 1)
          continue;

        // Process the results
        for(size_t i = 0; i < data.size (); i++)
        {
          if(processed[data[i]])
            continue;
          processed[data[i]] = true;
          copyPoint(host_cloud_->points[data[i]], point);
          queries_host.push_back (point);
          found_points++;
          r.indices.push_back(data[i]);
        }
      }

      // If number of queries is high enough do it here
      else
      {
        // Copy buffer
        // Move queries to GPU
        queries_device.create(queries_host.size());
        queries_device.upload(queries_host);
        // Execute search
        tree->radiusSearch(queries_device, tolerance, max_answers, result_device);
        // Copy results from GPU to Host
        result_device.sizes.download (sizes);
        result_device.data.download (data);
        // Store the previously found number of points
        previous_found_points = found_points;
        // Clear queries list
        queries_host.clear();
        
        for(size_t qp = 0; qp < sizes.size (); qp++)
        {
          for(int qp_r = 0; qp_r < sizes[qp]; qp_r++)
          {
            if(processed[data[qp_r + qp * max_answers]])
              continue;
            processed[data[qp_r + qp * max_answers]] = true;
            copyPoint(host_cloud_->points[data[qp_r + qp * max_answers]], point);
            queries_host.push_back (point);
            found_points++;
            r.indices.push_back(data[qp_r + qp * max_answers]);
          }
        }
      }
    }
    while (previous_found_points < found_points);
    
    // If this queue is satisfactory, add to the clusters
    if (found_points >= min_pts_per_cluster && found_points <= max_pts_per_cluster)
    {
      std::sort (r.indices.begin (), r.indices.end ());

      r.header = host_cloud_->header;
      clusters.push_back (r);   // We could avoid a copy by working directly in the vector
    }
  }
}

template <typename PointT> void 
pcl::gpu::EuclideanClusterExtraction<PointT>::extract (std::vector<pcl::PointIndices> &clusters)
{
/*
  // Initialize the GPU search tree
  if (!tree_)
  {
    tree_.reset (new pcl::gpu::Octree());
    ///@todo what do we do if input isn't a PointT cloud?
    tree_.setCloud(input_);
  }
*/
  if (!tree_->isBuilt())
  {
    tree_->build();
  }

  if(tree_->cloud_->size() != host_cloud_->points.size ())
  {
    // PCL_ERROR("[pcl::gpu::EuclideanClusterExtraction] size of host cloud and device cloud don't match!\n");
    std::cout << "[pcl::gpu::EuclideanClusterExtraction] size of host cloud and device cloud don't match!" << std::endl;
    return;
  }

  // Extract the actual clusters
  extractEuclideanClusters (host_cloud_, tree_, cluster_tolerance_, clusters, min_pts_per_cluster_, max_pts_per_cluster_, min_query_points_);
  std::cout << "INFO: end of extractEuclideanClusters " << std::endl;
  // Sort the clusters based on their size (largest one first)
  //std::sort (clusters.rbegin (), clusters.rend (), comparePointClusters);
}

#define PCL_INSTANTIATE_extractEuclideanClusters(T) template void PCL_EXPORTS pcl::gpu::extractEuclideanClusters (const boost::shared_ptr<pcl::PointCloud<T> >  &, const pcl::gpu::Octree::Ptr &,float, std::vector<PointIndices> &, unsigned int, unsigned int, int);
#define PCL_INSTANTIATE_EuclideanClusterExtraction(T) template class PCL_EXPORTS pcl::gpu::EuclideanClusterExtraction<T>;
#endif //PCL_GPU_SEGMENTATION_IMPL_EXTRACT_CLUSTERS_H_
