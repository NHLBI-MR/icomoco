
#include "util_functions.h"
#include <gadgetron/GadgetronTimer.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

namespace nhlbi_toolbox
{
    namespace utils
    {

        std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D(
            hoNDArray<float> *traj_dcw, int iSL, float enc_mat_z)
        {
            std::vector<size_t> dims = *traj_dcw->get_dimensions();
            std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); // Copy vector, but leave out first dim
            auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

            auto traj = boost::make_shared<hoNDArray<floatd3>>(reduced_dims);

            auto dcw_ptr = dcw->get_data_ptr();
            auto traj_ptr = traj->get_data_ptr();
            auto ptr = traj_dcw->get_data_ptr();

            // std::ofstream ofs("/tmp/traj_grad_flat.log");
            for (size_t i = 0; i < traj_dcw->get_number_of_elements() / 4; i++)
            {
                auto zencoding = float(-0.5 + iSL * 1 / enc_mat_z);

                traj_ptr[i][0] = ptr[i * 4];
                traj_ptr[i][1] = ptr[i * 4 + 1];
                traj_ptr[i][2] = zencoding;
                dcw_ptr[i] = ptr[i * 4 + 2]; // updated to 4 for ultimate legacy support
            }

            return std::make_tuple(traj, dcw);
        }

        template <unsigned int D>
        std::tuple<boost::shared_ptr<hoNDArray<vector_td<float, D>>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_all(
            hoNDArray<float> *traj_dcw, int iSL, float enc_mat_z)
        {
            std::vector<size_t> dims = *traj_dcw->get_dimensions();
            std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); // Copy vector, but leave out first dim
            auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

            auto traj = boost::make_shared<hoNDArray<vector_td<float, D>>>(reduced_dims);

            auto dcw_ptr = dcw->get_data_ptr();
            auto traj_ptr = traj->get_data_ptr();
            auto ptr = traj_dcw->get_data_ptr();

            // std::ofstream ofs("/tmp/traj_grad_flat.log");
            for (size_t i = 0; i < traj_dcw->get_number_of_elements() / (D + 1); i++)
            {
                if (D == 3)
                {
                    auto zencoding = float(-0.5 + iSL * 1 / enc_mat_z);

                    traj_ptr[i][0] = ptr[i * 4];
                    traj_ptr[i][1] = ptr[i * 4 + 1];
                    traj_ptr[i][2] = zencoding;
                    dcw_ptr[i] = ptr[i * 4 + 2]; // updated to 4 for ultimate legacy support
                }
                else
                {
                    traj_ptr[i][0] = ptr[i * 3];
                    traj_ptr[i][1] = ptr[i * 3 + 1];
                    dcw_ptr[i] = ptr[i * 3 + 2]; // updated to 4 for ultimate legacy support
                }
            }

            return std::make_tuple(traj, dcw);
        }
        // Convert to Templates to deal with view arrays etc.
        std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_2(
            hoNDArray<float> *traj_dcw, float enc_mat_z)
        {

            std::vector<size_t> dims = *traj_dcw->get_dimensions();
            std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); // Copy vector, but leave out first dim
            auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

            auto traj = boost::make_shared<hoNDArray<floatd2>>(reduced_dims);

            auto dcw_ptr = dcw->get_data_ptr();
            auto traj_ptr = traj->get_data_ptr();
            auto ptr = traj_dcw->get_data_ptr();
            // std::ofstream ofs("/tmp/traj_grad_flat.log");
            for (size_t i = 0; i < (traj_dcw->get_number_of_elements() / ((enc_mat_z > 1) ? 4 : 3)); i++)
            {
                traj_ptr[i][0] = ptr[i * ((enc_mat_z > 1) ? 4 : 3)];
                traj_ptr[i][1] = ptr[i * ((enc_mat_z > 1) ? 4 : 3) + 1];
                dcw_ptr[i] = ptr[i * ((enc_mat_z > 1) ? 4 : 3) + 2];
            }

            return std::make_tuple(traj, dcw);
        }
        // Convert to Templates to deal with view arrays etc.
        std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D_gen(
            hoNDArray<floatd2> *traj_input, hoNDArray<float> *dcw_input, int iSL, float enc_mat_z)
        {

            std::vector<size_t> dims = *traj_input->get_dimensions();
            auto dcw = boost::make_shared<hoNDArray<float>>(dims);

            auto traj = boost::make_shared<hoNDArray<floatd3>>(dims);

            auto dcw_ptr = dcw->get_data_ptr();
            auto traj_ptr = traj->get_data_ptr();
            auto ptr = traj_input->get_data_ptr();
            auto dcwptr = dcw_input->get_data_ptr();
            // std::ofstream ofs("/tmp/traj_grad_flat.log");
            for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
            {
                auto zencoding = float(-0.5 + iSL * 1 / enc_mat_z);

                traj_ptr[i][0] = ptr[i][0];
                traj_ptr[i][1] = ptr[i][1];
                traj_ptr[i][2] = zencoding;
                dcw_ptr[i] = dcwptr[i];
            }

            return std::make_tuple(traj, dcw);
        }

        template <typename T>
        cuNDArray<T> set_device(cuNDArray<T> *input, int device)
        {
            if (input->get_device() == device)
                return *input;

            int cur_device;
            if (cudaGetDevice(&cur_device) != cudaSuccess)
            {
                throw cuda_error("cuNDArray::set_device: unable to get device no");
            }

            if (cur_device != input->get_device() && cudaSetDevice(device) != cudaSuccess)
            {
                throw cuda_error("cuNDArray::set_device: unable to set device no");
            }

            cuNDArray<T> out(*input->get_dimensions(), device);

            if (cudaMemcpy(out.get_data_ptr(), input->get_data_ptr(), input->get_number_of_elements() * sizeof(T), cudaMemcpyDefault) != cudaSuccess)
            {
                cudaSetDevice(cur_device);
                throw cuda_error("cuNDArray::set_device: failed to copy data");
            }

            if (cudaSetDevice(cur_device) != cudaSuccess)
            {
                throw cuda_error("cuNDArray::set_device: unable to restore device to current device");
            }
            return out;
        }
        hoNDArray<floatd3> traj2grad(const hoNDArray<floatd3> &trajectory, float kspace_scaling)
        {
            auto gradients = trajectory;

            for (auto jj = 0; jj < trajectory.get_size(1); jj++)
            {
                auto traj_ptr = trajectory.get_data_ptr() + jj * trajectory.get_size(0);
                auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
                for (auto ii = 0; ii < trajectory.get_size(0); ii++)
                {
                    if (ii > 0)
                    {
                        grad_ptr[ii][0] = (traj_ptr[ii][0] - traj_ptr[ii - 1][0]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                        grad_ptr[ii][1] = (traj_ptr[ii][1] - traj_ptr[ii - 1][1]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                        grad_ptr[ii][2] = (traj_ptr[ii][2] - traj_ptr[ii - 1][2]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                    }
                }
            }
            return gradients;
        }

        hoNDArray<floatd2> traj2grad(const hoNDArray<floatd2> &trajectory, float kspace_scaling)
        {
            auto gradients = trajectory;

            for (auto jj = 0; jj < trajectory.get_size(1); jj++)
            {
                auto traj_ptr = trajectory.get_data_ptr() + jj * trajectory.get_size(0);
                auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
                for (auto ii = 0; ii < trajectory.get_size(0); ii++)
                {
                    if (ii > 0)
                    {
                        grad_ptr[ii][0] = (traj_ptr[ii][0] - traj_ptr[ii - 1][0]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                        grad_ptr[ii][1] = (traj_ptr[ii][1] - traj_ptr[ii - 1][1]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                    }
                }
            }
            return gradients;
        }
        hoNDArray<floatd2> traj2grad_3D2D(const hoNDArray<floatd3> &trajectory, float kspace_scaling)
        {
            auto gradients = hoNDArray<floatd2>(*trajectory.get_dimensions());
            gradients.fill(floatd2(0.0));
            for (auto jj = 0; jj < trajectory.get_size(1); jj++)
            {
                auto traj_ptr = trajectory.get_data_ptr() + jj * trajectory.get_size(0);
                auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
                for (auto ii = 0; ii < trajectory.get_size(0); ii++)
                {
                    if (ii > 0)
                    {
                        grad_ptr[ii][0] = (traj_ptr[ii][0] - traj_ptr[ii - 1][0]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                        grad_ptr[ii][1] = (traj_ptr[ii][1] - traj_ptr[ii - 1][1]) * 1000000 / (GAMMA * 10 * 2 * kspace_scaling);
                    }
                }
            }
            return gradients;
        }
        
        // template <typename T>
        // static inline double Lerp(T v0, T v1, T t)
        // {
        //     return (1 - t) * v0 + t * v1;
        // }

        // template <typename T>
        // static inline std::vector<T> Quantile(const std::vector<T> &inData, const std::vector<T> &probs)
        // {
        //     if (inData.empty())
        //     {
        //         return std::vector<T>();
        //     }

        //     if (1 == inData.size())
        //     {
        //         return std::vector<T>(1, inData[0]);
        //     }

        //     std::vector<T> data = inData;
        //     std::sort(data.begin(), data.end());
        //     std::vector<T> quantiles;

        //     for (size_t i = 0; i < probs.size(); ++i)
        //     {
        //         T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        //         size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        //         size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        //         T datLeft = data.at(left);
        //         T datRight = data.at(right);

        //         T quantile = Lerp<T>(datLeft, datRight, poi - left);

        //         quantiles.push_back(quantile);
        //     }

        //     return quantiles;
        // }

        void normalize_trajectory(
            hoNDArray<floatd2> *traj_input)
        {

            std::vector<size_t> dims = *traj_input->get_dimensions();
            auto ptr = traj_input->get_data_ptr();

            float max_x = 0;
            float min_x = 0;
            float max_y = 0;
            float min_y = 0;

            for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
            {
                if (max_x < ptr[i][0])
                    max_x = ptr[i][0];

                if (min_x > ptr[i][0])
                    min_x = ptr[i][0];

                if (max_y < ptr[i][1])
                    max_y = ptr[i][1];

                if (min_y > ptr[i][1])
                    min_y = ptr[i][1];
            }

            // auto normx = 2*std::max(std::abs(max_x),std::abs(min_x));
            // auto normy = 2*std::max(std::abs(max_y),std::abs(min_y));
            // for (size_t i = 0; i < traj_input->get_number_of_elements() / 3; i++)
            // {
            //     ptr[i][0]/=normx;
            //     ptr[i][1]/=normy;
            // }
        }
        cuNDArray<float_complext> estimateCoilmaps_slice(cuNDArray<float_complext> &data)
        {
            auto RO = data.get_size(0);
            auto E1 = data.get_size(1);
            auto E2 = data.get_size(2);
            auto CHA = data.get_size(3);

            data = permute(data, {0, 1, 3, 2});

            std::vector<size_t> recon_dims = {RO, E1, CHA}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> tempcsm(recon_dims);
            recon_dims.push_back(E2);
            cuNDArray<float_complext> csm(recon_dims);

            for (int iSL = 0; iSL < E2; iSL++)
            {
                cudaMemcpy(tempcsm.get_data_ptr(),
                           data.get_data_ptr() + RO * E1 * iSL * CHA,
                           RO * E1 * CHA * sizeof(float_complext), cudaMemcpyDeviceToDevice);

                cudaMemcpy(csm.get_data_ptr() + RO * E1 * iSL * CHA,
                           (estimate_b1_map<float, 2>(tempcsm)).get_data_ptr(),
                           RO * E1 * CHA * sizeof(float_complext), cudaMemcpyDeviceToDevice);
            }

            data = permute(data, {0, 1, 3, 2});

            return permute(csm, {0, 1, 3, 2});
        }
        float correlation(hoNDArray<float> a, hoNDArray<float> b)
        {
            float r = -1;

            float ma, mb;
            ma = Gadgetron::mean(&a);
            mb = Gadgetron::mean(&b);

            size_t N = a.get_number_of_elements();

            const float *pA = a.begin();
            const float *pB = b.begin();

            size_t n;

            double x(0), y(0), z(0);
            for (n = 0; n < N; n++)
            {
                x += (pA[n] - ma) * (pA[n] - ma);
                y += (pB[n] - mb) * (pB[n] - mb);
                z += (pA[n] - ma) * (pB[n] - mb);
            }

            double p = std::sqrt(x * y);
            if (p > 0)
            {
                r = (float)(z / p);
            }
            return r;
        }
        void attachHeadertoImageArray(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, const ISMRMRD::IsmrmrdHeader &h)
        {

            int n = 0;
            int s = 0;
            int loc = 0;
            std::vector<size_t> header_dims(3);
            header_dims[0] = 1;
            header_dims[1] = 1;
            header_dims[2] = 1;
            imarray.headers_.create(header_dims);
            imarray.meta_.resize(1);

            auto fov = h.encoding.front().encodedSpace.fieldOfView_mm;
            auto val = h.encoding.front().encodingLimits.kspace_encoding_step_0;
            imarray.headers_(n, s, loc).matrix_size[0] = h.encoding.front().encodedSpace.matrixSize.x;
            imarray.headers_(n, s, loc).matrix_size[1] = h.encoding.front().encodedSpace.matrixSize.y;
            imarray.headers_(n, s, loc).matrix_size[2] = h.encoding.front().reconSpace.matrixSize.z;
            imarray.headers_(n, s, loc).field_of_view[0] = fov.x;
            imarray.headers_(n, s, loc).field_of_view[1] = fov.y;
            imarray.headers_(n, s, loc).field_of_view[2] = fov.z;
            imarray.headers_(n, s, loc).channels = 1;
            imarray.headers_(n, s, loc).average = acqhdr.idx.average;
            imarray.headers_(n, s, loc).slice = acqhdr.idx.slice;
            imarray.headers_(n, s, loc).contrast = acqhdr.idx.contrast;
            imarray.headers_(n, s, loc).phase = acqhdr.idx.phase;
            imarray.headers_(n, s, loc).repetition = acqhdr.idx.repetition;
            imarray.headers_(n, s, loc).set = acqhdr.idx.set;
            imarray.headers_(n, s, loc).acquisition_time_stamp = acqhdr.acquisition_time_stamp;
            imarray.headers_(n, s, loc).position[0] = acqhdr.position[0];
            imarray.headers_(n, s, loc).position[1] = acqhdr.position[1];
            imarray.headers_(n, s, loc).position[2] = acqhdr.position[2];
            imarray.headers_(n, s, loc).read_dir[0] = acqhdr.read_dir[0];
            imarray.headers_(n, s, loc).read_dir[1] = acqhdr.read_dir[1];
            imarray.headers_(n, s, loc).read_dir[2] = acqhdr.read_dir[2];
            imarray.headers_(n, s, loc).phase_dir[0] = acqhdr.phase_dir[0];
            imarray.headers_(n, s, loc).phase_dir[1] = acqhdr.phase_dir[1];
            imarray.headers_(n, s, loc).phase_dir[2] = acqhdr.phase_dir[2];
            imarray.headers_(n, s, loc).slice_dir[0] = acqhdr.slice_dir[0];
            imarray.headers_(n, s, loc).slice_dir[1] = acqhdr.slice_dir[1];
            imarray.headers_(n, s, loc).slice_dir[2] = acqhdr.slice_dir[2];
            imarray.headers_(n, s, loc).patient_table_position[0] = acqhdr.patient_table_position[0];
            imarray.headers_(n, s, loc).patient_table_position[1] = acqhdr.patient_table_position[1];
            imarray.headers_(n, s, loc).patient_table_position[2] = acqhdr.patient_table_position[2];
            imarray.headers_(n, s, loc).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
            imarray.headers_(n, s, loc).image_index = 1;
        }

        void filterImagealongSlice(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma)
        {
            hoNDArray<std::complex<float>> fx(image.get_size(0)),
                fy(image.get_size(1)),
                fz(image.get_size(2));

            generate_symmetric_filter(image.get_size(0), fx, ISMRMRD_FILTER_NONE);
            generate_symmetric_filter(image.get_size(1), fy, ISMRMRD_FILTER_NONE);
            generate_symmetric_filter(image.get_size(2), fz, ftype, fsigma, (size_t)std::ceil(fwidth * image.get_size(2)));

            hoNDArray<std::complex<float>> fxyz(image.get_dimensions());
            compute_3d_filter(fx, fy, fz, fxyz);

            Gadgetron::hoNDFFT<float>::instance()->ifft3c(image);
            multiply(&image, &fxyz, &image);
            Gadgetron::hoNDFFT<float>::instance()->fft3c(image);
        }

        template <typename T>
        hoNDArray<T> convolve(hoNDArray<T> &input, hoNDArray<T> &kernel)
        {
            using namespace std;
            hoNDArray<T> output(input.dimensions());
            auto dims = from_std_vector<size_t, 1>(input.dimensions());

            int kernel_size = kernel.size();
            vector_td<int, 1> index;

            long long dim_len = dims[0];

#pragma omp parallel for shared(kernel, input, output) private(index)
            for (int x = 0; x < dims[0]; x++)
            {
                index[0] = x;
                T summation = T(0);
                for (int k = 0; k < kernel_size; k++)
                {
                    int kl = k - kernel_size / 2 + int(index[0]);
                    if (kl >= 0 && kl < input.get_size(0))
                    {
                        long long offset = (kl - x);
                        summation += kernel[k] * input[x + offset];
                    }
                }
                output[x] = summation;
            }

            return output;
        }
        template <typename T>
        std::vector<size_t> sort_indexes(std::vector<T> &v)
        {

            // initialize original index locations
            std::vector<size_t> idx(v.size());
            std::iota(idx.begin(), idx.end(), 0);

            // sort indexes based on comparing values in v
            // using std::stable_sort instead of std::sort
            // to avoid unnecessary index re-orderings
            // when v contains elements of equal values
            std::stable_sort(idx.begin(), idx.end(),
                             [&v](size_t i1, size_t i2)
                             { return v[i1] <= v[i2]; });

            return idx;
        }
        template <typename T>
        hoNDArray<T> paddedCovolution(hoNDArray<T> &input, hoNDArray<T> &kernel)
        {
            auto paddedRect = nhlbi_toolbox::utils::padForConv(input);

            auto temp = convolve(paddedRect, kernel);

            hoNDArray<T> output(input.get_size(0));
#pragma omp parallel for
            for (auto i = 0; i < input.get_size(0); i++)
                output(i) = temp(i + input.get_size(0));

            return output;
        }
        void filterImage(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma)
        {
            hoNDArray<std::complex<float>> fx(image.get_size(0)),
                fy(image.get_size(1)),
                fz(image.get_size(2));

            generate_symmetric_filter(image.get_size(0), fx, ftype, fsigma, size_t(std::ceil(image.get_size(0) * fwidth)));
            generate_symmetric_filter(image.get_size(1), fy, ftype, fsigma, size_t(std::ceil(image.get_size(0) * fwidth)));
            generate_symmetric_filter(image.get_size(2), fz, ISMRMRD_FILTER_NONE, fsigma, fwidth);

            hoNDArray<std::complex<float>> fxyz(image.get_dimensions());
            compute_3d_filter(fx, fy, fz, fxyz);

            Gadgetron::hoNDFFT<float>::instance()->ifft3c(image);
            multiply(&image, &fxyz, &image);
            Gadgetron::hoNDFFT<float>::instance()->fft3c(image);
        }

        template <typename T>
        void write_gpu_nd_array(cuNDArray<T> &data, std::string filename)
        {
            boost::shared_ptr<hoNDArray<T>> data_host = data.to_host();
            write_nd_array<T>(data_host.get(), filename.c_str());
        }

        template <typename T>
        void write_cpu_nd_array(hoNDArray<T> &data, std::string filename)
        {
            auto d = &data;
            write_nd_array<T>(d, filename.c_str());
        }

        template <typename T>
        cuNDArray<T> concat(std::vector<cuNDArray<T>> &arrays)
        {
            if (arrays.empty())
                return cuNDArray<T>();

            const cuNDArray<T> &first = *std::begin(arrays);

            auto dims = first.dimensions();
            auto size = first.size();

            if (!std::all_of(begin(arrays), end(arrays), [&](auto &array)
                             { return dims == array.dimensions(); }) ||
                !std::all_of(begin(arrays), end(arrays), [&](auto &array)
                             { return size == array.size(); }))
            {
                throw std::runtime_error("Array size or dimensions do not match.");
            }

            dims.push_back(arrays.size());
            cuNDArray<T> output(dims);

            auto slice_dimensions = output.dimensions();
            slice_dimensions.pop_back();

            size_t stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1, std::multiplies<size_t>());

            for (size_t iter = 0; iter < arrays.size(); iter++)
            {
                cudaMemcpy(output.get_data_ptr() + iter * stride,
                           arrays.at(iter).get_data_ptr(),
                           stride * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            return output;
        }

        template <typename T>
        hoNDArray<T> padForConv(hoNDArray<T> &input)
        {
            // create output [input(end:-1:1) input input(end:-1:1)]
            hoNDArray<T> output(input.get_size(0) * 3);
#pragma omp parallel for
            for (auto i = 0; i < input.get_size(0); i++)
            {
                output(i) = input((input.get_size(0) - 1) - i);
                output(i + input.get_size(0)) = input(i);
                output(i + input.get_size(0) * 2) = input((input.get_size(0) - 1) - i);
            }

            return output;
        }
        template <typename T>
        void normalizeImages(hoNDArray<T> &input_image)
        {
            GDEBUG("Normalizing Images");

            auto max = input_image[0];
            auto min = input_image[0];
#pragma omp parallel for
            for (unsigned long int i = 0; i < input_image.get_number_of_elements(); i++)
            {
                if (std::abs(input_image[i]) > std::abs(max))
                    max = input_image[i];

                if (std::abs(input_image[i]) < std::abs(min))
                    min = input_image[i];
            }
#pragma omp parallel for
            for (unsigned long int i = 0; i < input_image.get_number_of_elements(); i++)
            {
                input_image[i] -= min;
                input_image[i] /= max;
            }
        }
        template <typename T>
        std::vector<T> sliceVec(std::vector<T> &v, int start, int end, int stride)
        {
            int oldlen = v.size();
            int newlen;

            if (end == -1 or end >= oldlen)
            {
                newlen = (oldlen - start) / stride;
            }
            else
            {
                newlen = (end - start) / stride;
            }

            std::vector<T> nv(newlen);

            for (int i = 0; i < newlen; i++)
            {
                nv[i] = v[start + i * stride];
            }
            return nv;
        }

        int selectCudaDevice()
        {
            int totalNumberofDevice = cudaDeviceManager::Instance()->getTotalNumberOfDevice();
            int selectedDevice = 0;
            size_t freeMemory = 0;

            for (int dno = 0; dno < totalNumberofDevice; dno++)
            {
                cudaSetDevice(dno);
                if (cudaDeviceManager::Instance()->getFreeMemory(dno) > freeMemory)
                {
                    freeMemory = cudaDeviceManager::Instance()->getFreeMemory(dno);
                    selectedDevice = dno;
                }
            }
            // GDEBUG_STREAM("Selected Device: " << selectedDevice);
            return selectedDevice;
        }

        std::vector<int> FindCudaDevices(unsigned long req_size)
        {
            int totalNumberofDevice = cudaDeviceManager::Instance()->getTotalNumberOfDevice();
            size_t freeMemory = 0;
            std::vector<int> gpus;
            struct cudaDeviceProp properties;

            /* machines with no GPUs can still report one emulation device */

            // GDEBUG_STREAM("req_size# " << req_size);

            if (req_size < (long)6 * std::pow(1024, 3))
                req_size = (long)6 * std::pow(1024, 3);

            // GDEBUG_STREAM("req_size# " << req_size);

            for (int dno = 0; dno < totalNumberofDevice; dno++)
            {
                cudaSetDevice(dno);
                //GDEBUG_STREAM("Free_memory# " << cudaDeviceManager::Instance()->getFreeMemory(dno));
                cudaGetDeviceProperties(&properties, dno);
                //  GDEBUG_STREAM("MajorMode# " << properties.major);
                //  GDEBUG_STREAM("Minor# " << properties.minor);

                if (cudaDeviceManager::Instance()->getFreeMemory(dno) > req_size && properties.major >= 6)
                {
                    gpus.push_back(dno);
                }
            }

            return gpus;
        }

        void enable_peeraccess()
        {
            int totalNumberofDevice = cudaDeviceManager::Instance()->getTotalNumberOfDevice();
            for (int dno = 0; dno < totalNumberofDevice; dno++)
            {
                cudaSetDevice(dno);
                for (int dn1 = 0; dn1 < totalNumberofDevice; dn1++)
                {
                    int flag;
                    cudaDeviceCanAccessPeer(&flag, dno, dn1);
                    if (!flag)
                        cudaDeviceEnablePeerAccess(dn1, 0);
                }
            }
        }

        void setNumthreadsonGPU(int Number)
        {
            omp_set_num_threads(Number);
            int id = omp_get_num_threads();
        }

        std::vector<hoNDArray<float>> estimateDCF_slice(std::vector<std::vector<hoNDArray<floatd3>>> trajectories, std::vector<std::vector<hoNDArray<float>>> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                                        std::vector<size_t> image_dims_, bool fullySampled)
        {
            boost::shared_ptr<cuNDArray<float>> dcw;
            boost::shared_ptr<cuNDArray<floatd3>> traj;
            boost::shared_ptr<cuNDArray<float>> tdcf;

            std::vector<hoNDArray<float>> dcw_vec;
#pragma omp parallel
            omp_set_num_threads(cudaDeviceManager::Instance()->getTotalNumberOfDevice() - 1);
            int id = omp_get_num_threads();
#pragma omp for

            for (auto ii = 0; ii < trajectories.size(); ii++)
            {
                if (!fullySampled || ii == 0)
                {
                    cudaSetDevice(nhlbi_toolbox::utils::selectCudaDevice());

                    auto temp = Gadgetron::concat(trajectories[ii]);
                    auto temp_dcf = Gadgetron::concat(dcf[ii]);

                    tdcf = boost::make_shared<cuNDArray<float>>(temp_dcf);
                    traj = boost::make_shared<cuNDArray<floatd3>>(temp);
                    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
                    cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
                    traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

                    auto t = *(Gadgetron::estimate_dcw<float, 3>((*traj),
                                                                 *tdcf,
                                                                 from_std_vector<size_t, 3>(image_dims_),
                                                                 oversampling_factor_,
                                                                 size_t(iterations), 5.5, ConvolutionType::ATOMIC));

                    // float scale_factor = float(image_dims_os_[0]*image_dims_os_[1]*image_dims_os_[2]) / asum((&t));
                    // t *= scale_factor;
                    dcw_vec.push_back(*(t.to_host()));
                }
                else
                {
                    // auto temp = dcw_vec.at(ii-1);
                    dcw_vec.push_back(dcw_vec.at(ii - 1));
                }
            }
            return dcw_vec;
        }

        hoNDArray<float> estimateDCF(hoNDArray<floatd3> trajectories, hoNDArray<float> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                     std::vector<size_t> image_dims_, bool fullySampled)
        {
            GadgetronTimer timer("DCF : ");

            boost::shared_ptr<cuNDArray<float>> dcw;
            boost::shared_ptr<cuNDArray<floatd3>> traj;
            boost::shared_ptr<cuNDArray<float>> tdcf;

            // std::vector<hoNDArray<float>> dcw_vec;

            cudaSetDevice(nhlbi_toolbox::utils::selectCudaDevice());

            // auto temp = nhlbi_toolbox::utils::concat<floatd3>(trajectories);
            // auto temp_dcf = nhlbi_toolbox::utils::concat<float>(dcf);

            tdcf = boost::make_shared<cuNDArray<float>>(dcf);
            traj = boost::make_shared<cuNDArray<floatd3>>(trajectories);
            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
            traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

            auto t = *(Gadgetron::estimate_dcw<float, 3>((*traj),
                                                         *tdcf,
                                                         from_std_vector<size_t, 3>(image_dims_),
                                                         oversampling_factor_,
                                                         size_t(iterations), 5.5, ConvolutionType::ATOMIC));

            // float scale_factor = float(image_dims_os_[0]*image_dims_os_[1]*image_dims_os_[2]) / asum((&t));
            // t *= scale_factor;
            // dcw_vec.push_back(*(t.to_host()));
            auto output = *(t.to_host());
            return output;
        }

        template <template <class> class ARRAY>
        void
        set_data(ARRAY<float_complext> &sp_data, ARRAY<floatd3> &sp_traj, ARRAY<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                 boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, std::vector<size_t> indata_dims, std::vector<size_t> outdata_dims, int currDev)
        {
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, sp_data.get_data_ptr());

            cuData = boost::make_shared<cuNDArray<float_complext>>((outdata_dims));

            auto stride = std::accumulate(outdata_dims.begin(), outdata_dims.end() - 1, size_t(1), std::multiplies<size_t>());

            // use outdata_dims because it should be RO INT and CHA always
            if (outdata_dims[2] != indata_dims[2]) // means not permuted
            {
                for (auto ii = 0; ii < outdata_dims[1]; ii++)
                {
                    for (size_t iCHA = 0; iCHA < outdata_dims[2]; iCHA++)
                    {

                        cudaMemcpy(cuData.get()->get_data_ptr() + stride * iCHA + outdata_dims[0] * ii,
                                   sp_data.get_data_ptr() + indata_dims[0] * iCHA + indata_dims[0] * indata_dims[1] * ii,
                                   indata_dims[0] * sizeof(complext<float>), cudaMemcpyDefault);
                    }
                }
            }
            else
            {
                cuData = boost::make_shared<cuNDArray<float_complext>>((sp_data));
            }
            cuData.get()->squeeze();

            traj = boost::make_shared<cuNDArray<floatd3>>((sp_traj));

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
            traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);
            dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            cuNDArray<float> flat_dcw(flat_dims, dcw.get()->get_data_ptr());
            dcw = boost::make_shared<cuNDArray<float>>(flat_dcw);
            // if (~useIterativeDCW)
            // {
            //     dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            // }
            // if (useIterativeDCWEstimated && useIterativeDCW)
            // {
            //     dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            //     cuNDArray<float> flat_dcw(flat_dims, dcw->get_data_ptr());
            //     dcw = boost::make_shared<cuNDArray<float>>(flat_dcw);
            // }

            // cuNDArray<float> flat_dcw(flat_dims, dcw.get_data_ptr());
            // dcw = cuNDArray<float>(flat_dcw);
        }
        std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj)
        {
            auto sl_traj = arma::vec(sp_traj.get_size(1));
            for (auto ii = 0; ii < sp_traj.get_size(1); ii++)
                sl_traj[ii] = sp_traj(1, ii)[2];

            auto z_encodes = arma::vec(arma::unique(sl_traj));
            // z_encodes.print();

            std::vector<arma::uvec> slice_indexes;
            for (auto ii = 0; ii < z_encodes.n_elem; ii++)
            {
                arma::uvec temp = (find(sl_traj == z_encodes[ii]));
                slice_indexes.push_back(temp);
                //  slice_indexes[ii].print();
            }

            return slice_indexes;
        }

        arma::fmat33 lookup_PCS_DCS(std::string PCS_description)
        {
            arma::fmat33 A;

            if (PCS_description == "HFP")
            {
                A(0, 0) = 0;
                A(0, 1) = 1;
                A(0, 2) = 0;
                A(1, 0) = -1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFS")
            {
                A(0, 0) = 0;
                A(0, 1) = -1;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFDR")
            {
                A(0, 0) = 1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = 1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFDL")
            {
                A(0, 0) = -1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFP")
            {
                A(0, 0) = 0;
                A(0, 1) = 1;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFS")
            {
                A(0, 0) = 0;
                A(0, 1) = -1;
                A(0, 2) = 0;
                A(1, 0) = -1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFDR")
            {
                A(0, 0) = 1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = -1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFDL")
            {
                A(0, 0) = -1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = 1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }

            return A;
        } // namespace corrections

        template <class T>
        hoNDArray<T> mean_complex(hoNDArray<T> input, unsigned int dim)
        {
            auto real_a = real(input);
            auto imag_a = imag(input);

            auto mean_r = sum(real_a, dim);
            auto mean_i = sum(imag_a, dim);

            mean_r /= real_a.get_size(dim);
            mean_i /= imag_a.get_size(dim);

            auto output = *real_imag_to_complex<T>(&mean_r, &mean_i);
            return (output);
        }

        template <class T>
        hoNDArray<T> std_complex(hoNDArray<T> input, unsigned int dim)
        {
            auto real_a = real(input);
            auto imag_a = imag(input);

            auto std_r = std_real<float>(real_a, dim);
            auto std_i = std_real<float>(imag_a, dim);

            auto output = *real_imag_to_complex<T>(&std_r, &std_i);
            return (output);
        }
        template <class T>
        hoNDArray<T> std_real(hoNDArray<T> input, unsigned int dim)
        {
            auto dims = *input.get_dimensions();
            auto num_elements = std::accumulate(dims.begin(), dims.end() - 1, size_t(1), std::multiplies<size_t>());

            hoNDArray<T> std_dev_num(*input.get_dimensions());
            auto mean_calc = sum(input, dim);
            mean_calc /= input.get_size(dim);

            for (auto ii = 0; ii < input.get_size(dim) - 1; ii++)
            {
                // std::set_difference(ho_noise_images.begin() + ii * num_elements, ho_noise_images.begin() + (ii + 1) * num_elements - 1, mean_cplx.begin(), mean_cplx.end(),
                // std::inserter(std_dev_num.begin() + ii * num_elements, std_dev_num.begin() + (ii + 1) * num_elements - 1));
                std::transform(input.begin() + ii * num_elements, input.begin() + (ii + 1) * num_elements - 1, mean_calc.begin(), std_dev_num.begin() + ii * num_elements, std::minus<T>());
            }

            square_inplace(&std_dev_num);
            auto std_sum = *sum<T>(&std_dev_num, dim);
            std_sum /= float(input.get_size(dim) - 1);
            sqrt_inplace(&std_sum);
            return std_sum;
        }

        template hoNDArray<float> std_real(hoNDArray<float> input, unsigned int dim);
        template hoNDArray<float_complext> std_complex(hoNDArray<float_complext> input, unsigned int dim);
        template hoNDArray<std::complex<float>> std_complex(hoNDArray<std::complex<float>> input, unsigned int dim);

        template void set_data(cuNDArray<float_complext> &sp_data, cuNDArray<floatd3> &sp_traj, cuNDArray<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                               boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, std::vector<size_t> indata_dims, std::vector<size_t> outdata_dims, int currDev);
        template void set_data(hoNDArray<float_complext> &sp_data, hoNDArray<floatd3> &sp_traj, hoNDArray<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                               boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, std::vector<size_t> indata_dims, std::vector<size_t> outdata_dims, int currDev);

        template void write_gpu_nd_array(cuNDArray<float> &data, std::string filename);
        template void write_gpu_nd_array(cuNDArray<float_complext> &data, std::string filename);
        template void write_gpu_nd_array(cuNDArray<floatd3> &data, std::string filename);
        template void write_gpu_nd_array(cuNDArray<floatd2> &data, std::string filename);

        template void write_cpu_nd_array(hoNDArray<float> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<float_complext> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<floatd3> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<floatd2> &data, std::string filename);

        template void write_cpu_nd_array(hoNDArray<complex_float_t> &data, std::string filename);

        template cuNDArray<float_complext> concat(std::vector<cuNDArray<float_complext>> &arrays);
        template cuNDArray<floatd3> concat(std::vector<cuNDArray<floatd3>> &arrays);
        template cuNDArray<floatd2> concat(std::vector<cuNDArray<floatd2>> &arrays);
        template cuNDArray<float> concat(std::vector<cuNDArray<float>> &arrays);

        // template hoNDArray<float_complext> concat(std::vector<hoNDArray<float_complext>> &arrays);
        // template hoNDArray<floatd3> concat(std::vector<hoNDArray<floatd3>> &arrays);
        // template hoNDArray<floatd2> concat(std::vector<hoNDArray<floatd2>> &arrays);
        // template hoNDArray<float> concat(std::vector<hoNDArray<float>> &arrays);

        template std::vector<cuNDArray<float>> sliceVec(std::vector<cuNDArray<float>> &v, int start, int end, int stride);
        template std::vector<cuNDArray<floatd2>> sliceVec(std::vector<cuNDArray<floatd2>> &v, int start, int end, int stride);
        template std::vector<cuNDArray<float_complext>> sliceVec(std::vector<cuNDArray<float_complext>> &v, int start, int end, int stride);
        template std::vector<ISMRMRD::AcquisitionHeader> sliceVec(std::vector<ISMRMRD::AcquisitionHeader> &v, int start, int end, int stride);

        template hoNDArray<std::complex<float>> padForConv(hoNDArray<std::complex<float>> &input);
        template hoNDArray<float_complext> padForConv(hoNDArray<float_complext> &input);
        template hoNDArray<float> padForConv(hoNDArray<float> &input);

        template hoNDArray<float_complext> convolve(hoNDArray<float_complext> &input, hoNDArray<float_complext> &kernel);
        template hoNDArray<std::complex<float>> convolve(hoNDArray<std::complex<float>> &input, hoNDArray<std::complex<float>> &kernel);

        template hoNDArray<std::complex<float>> paddedCovolution(hoNDArray<std::complex<float>> &input, hoNDArray<std::complex<float>> &kernel);

        template std::vector<size_t> sort_indexes(std::vector<int> &v);
        template std::vector<size_t> sort_indexes(std::vector<float> &v);
        template std::vector<size_t> sort_indexes(std::vector<uint32_t> &v);

        template void normalizeImages(hoNDArray<float> &input_image);
        template void normalizeImages(hoNDArray<std::complex<float>> &input_image);

        template std::tuple<boost::shared_ptr<hoNDArray<vector_td<float, 2>>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_all(
            hoNDArray<float> *traj_dcw, int iSL, float enc_mat_z);

        template std::tuple<boost::shared_ptr<hoNDArray<vector_td<float, 3>>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_all(
            hoNDArray<float> *traj_dcw, int iSL, float enc_mat_z);

        template cuNDArray<float> set_device(cuNDArray<float> *input, int device);
        template cuNDArray<float_complext> set_device(cuNDArray<float_complext> *input, int device);
        template cuNDArray<floatd3> set_device(cuNDArray<floatd3> *input, int device);
        template cuNDArray<floatd2> set_device(cuNDArray<floatd2> *input, int device);

        template hoNDArray<float_complext> mean_complex(hoNDArray<float_complext> input, unsigned int dim);
        template hoNDArray<std::complex<float>> mean_complex(hoNDArray<std::complex<float>> input, unsigned int dim);
    } // namespace utils
} // namespace nhlbi_toolbox
