#include "common.h"
#include "FaceRecognition.h"
#include <boost/format.hpp> 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <boost/algorithm/string.hpp>    

FaceRecognition::FaceRecognition()
{
}

FaceRecognition::~FaceRecognition()
{
}

void FaceRecognition::SimpleApproach()
{
	using namespace cv;
	using namespace std;
	boost::format fmt("./%s%d.%s");
	vector<Mat> imgs;
	vector<Mat> face_vectors;
	string dir = "F:/Datasets/LFW/samples/";
	string name = "Mark_Philippoussis_0001.jpg";
	string path = dir + name;
	int num = 11;
	for (int i = 1; i < num + 1; i++)
	{
		if (i < 10)
		{
			name = (fmt%"Mark_Philippoussis_000" % (i) % "jpg").str();
		}
		else if (i >= 10 && i < 100)
		{
			name = (fmt%"Mark_Philippoussis_00" % (i) % "jpg").str();
		}
		path = dir + name;
		Mat img;
		img = imread(path, 0);
		imgs.push_back(img);
		int H = img.rows;
		int W = img.cols;
		Mat face_vector = Mat_<double>(H*W, 1);
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				face_vector.at<double>(H*i + j, 0) = img.at<uchar>(i, j);
			}
		}
		face_vectors.push_back(face_vector);
	}

	vector<Mat> imgs_test;
	vector<Mat> face_vectors_test;
	num = 3;
	for (int i = 1; i < num + 1; i++)
	{
		name = (fmt%"Vaclav_Havel_000" % (i) % "jpg").str();
		path = dir + name;
		Mat img;
		img = imread(path, 0);
		imgs_test.push_back(img);
		int H = img.rows;
		int W = img.cols;
		Mat face_vector_test = Mat_<double>(H*W, 1);
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				face_vector_test.at<double>(H*i + j, 0) = img.at<uchar>(i, j);
			}
		}
		face_vectors_test.push_back(face_vector_test);
	}
	vector<double> errs;
	int H = face_vectors_test[0].rows;
	int W = face_vectors_test[0].cols;
	for (int i = 0; i < face_vectors_test.size(); i++)
	{
		for (int j = 0; j < face_vectors.size(); j++)
		{
			double err = 0;
			for (int k = 0; k < H; k++)
			{
				err += abs(face_vectors_test[i].at<double>(k) - face_vectors[j].at<double>(k));
			}
			errs.push_back(err);
		}
	}
	cout << errs[11 * 0 + 0] << endl;
}

void FaceRecognition::Explore()
{
	using namespace cv;
	using namespace std;
	string dir = "F:/Datasets/LFW/samples/";
	string name = "Mark_Philippoussis_0001.jpg";
	string path = dir + name;
	Mat img = imread(path, 0);
	img.convertTo(img, CV_32FC1);
	int H = img.rows;
	int W = img.cols;
	Mat img_filtered = img.clone();
	Mat kernel = img(Rect(Point(48, 60), Point(65, 73))).clone();
	int kernel_W = 65 - 48;
	int kernel_H = 73 - 60;
	//filter2D(img, img_filtered,-1,kernel);
	//img(Rect(Point(48, 60), Point(65, 73)))=img(Rect(Point(48, 60), Point(65, 73))) - kernel;
	//for (int i = 0; i+ kernel_H < H; i+= kernel_H)
	//{
	//    for (int j = 0; j+ kernel_W < W; j+= kernel_W)
	//    {
	//        img_filtered(Rect(j, i, kernel_W, kernel_H))=img(Rect(j, i, kernel_W, kernel_H)) - kernel;
	//    }
	//}

	for (int i = 0; i + kernel_H < H; i++)
	{
		for (int j = 0; j + kernel_W < W; j++)
		{
			float tot = 0.0;
			Mat tmp = img(Rect(j, i, kernel_W, kernel_H)) - kernel;
			for (int ii = 0; ii < kernel_H; ii++)
			{
				for (int jj = 0; jj < kernel_W; jj++)
				{
					tot += abs(tmp.at<float>(ii, jj));
				}
			}
			img_filtered(Rect(j, i, 1, 1)) = tot;
		}
	}

	int layer_num = 8;
	PCA pca(img, Mat(), CV_PCA_DATA_AS_COL, layer_num);
	Mat eigenvalues = pca.eigenvalues;
	Mat eigenvectors = pca.eigenvectors;
	cout << pca.mean.size() << endl;
	cout << pca.eigenvalues.size() << endl;
	cout << pca.eigenvectors.size() << endl;
	Mat dst = pca.project(img);
	Mat src = pca.backProject(dst);

	layer_num = 3;
	Mat _33 = (Mat_<double>(3, 3) <<
		-1, 2, 0,
		0, 3, 4,
		0, 0, 7);
	PCA pca_33(_33, Mat(), CV_PCA_DATA_AS_COL, layer_num);
	eigenvalues = pca_33.eigenvalues;
	eigenvectors = pca_33.eigenvectors;

	Eigen::Matrix3d matrix;
	matrix <<
		-1, 2, 0,
		0, 3, 4,
		0, 0, 7;
	matrix.eigenvalues();
	Eigen::EigenSolver<Eigen::Matrix3d> eig(matrix);
	eig.eigenvalues();
	eig.eigenvectors();
	eig.pseudoEigenvalueMatrix();
	eig.pseudoEigenvectors();
}

void FaceRecognition::EigenFaces_explore()
{
	using namespace cv;
	using namespace std;
	boost::format fmt("./%s%d.%s");
	vector<Mat> imgs;
	vector<Mat> face_vectors;
	string dir = "F:/Datasets/LFW/lfw-deepfunneled/Aaron_Peirsol/";
	string name = "Aaron_Peirsol_0001.jpg";
	string path = dir + name;
	int num = 4;
	Mat img;
	img = imread(path, 0);
	int H = img.rows;
	int W = img.cols;
	Mat img_test(Size(W, H), CV_8UC1);
	Mat face_U(Size(num, H*W), CV_32FC1);
	for (int i = 0; i < num; i++)
	{
		if (i < 9)
		{
			name = (fmt%"Aaron_Peirsol_000" % (i + 1) % "jpg").str();
		}
		else if (i >= 9 && i < 99)
		{
			name = (fmt%"Aaron_Peirsol_00" % (i + 1) % "jpg").str();
		}
		path = dir + name;
		img = imread(path, 0);
		imgs.push_back(img);
		int H = img.rows;
		int W = img.cols;
		for (int j = 0; j < H; j++)
		{
			for (int k = 0; k < W; k++)
			{
				face_U.at<float>(W*j + k, i) = img.at<uchar>(j, k);
				img_test.at<uchar>(j, k) = face_U.at<float>(W*j + k, i);
			}
		}
	}
	//Mat face_L = face_U * face_U.t();
	Mat face_C = face_U.t() * face_U;
	H = face_C.rows;
	W = face_C.cols;
	Eigen::Matrix<double, 4, 4> matrix;
	for (int i = 0; i < num; i++)
	{
		for (int j = 0; j < W; j++)
		{
			matrix(i, j) = face_C.at<float>(i, j);
		}
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(matrix);
	cout << eig.eigenvalues() << endl;
	cout << eig.eigenvectors() << endl;
	Mat row_picked = face_U.row(10000).t();
	vector<double> coeffs_C(W, 0);
	Mat eigen_vectors_C = Mat::zeros(Size(W, H), CV_32FC1);
	for (int i = 0; i < W; i++)
	{
		Mat eigen_vector(Size(1, H), CV_32FC1);
		for (int j = 0; j < H; j++)
		{
			eigen_vector.at<float>(j, 0) = eig.eigenvectors().block(0, i, 4, 1)(j, 0);
		}
		normalize(eigen_vector, eigen_vector);
		eigen_vector.copyTo(eigen_vectors_C.col(i));
		Mat coeff = eigen_vector.t()*row_picked;
		coeffs_C[i] = coeff.at<float>(0, 0);
	}
	Mat row_picked_estimate = Mat::zeros(Size(1, H), CV_32FC1);
	for (int i = 0; i < W; i++)
	{
		row_picked_estimate += coeffs_C[i] * eigen_vectors_C.col(i);
	}
	Mat eigen_vectors_L = face_U * eigen_vectors_C;
	for (int i = 0; i < W; i++)
	{
		normalize(eigen_vectors_L.col(i), eigen_vectors_L.col(i));
	}
	Mat eigen_vectors_L_0 = eigen_vectors_L.col(0);
	Mat eigen_vectors_L_1 = eigen_vectors_L.col(1);
	Mat res = eigen_vectors_L_0.t()*eigen_vectors_L_1;
	Mat res1 = eigen_vectors_L_0.t()*eigen_vectors_L_0;

	int img_H = img.rows;
	int img_W = img.cols;
	Mat col_picked_estimate0 = Mat::zeros(Size(img_W, img_H), CV_32FC1);
	Mat col_picked_estimate1 = Mat::zeros(Size(img_W, img_H), CV_32FC1);
	Mat col_picked_estimate2 = Mat::zeros(Size(img_W, img_H), CV_32FC1);
	Mat col_picked_estimate3 = Mat::zeros(Size(img_W, img_H), CV_32FC1);
	Mat col_picked_estimate_L = Mat::zeros(Size(img_W, img_H), CV_32FC1);
	PCAImage(img, face_C, face_U, eigen_vectors_L, col_picked_estimate0, 0);
	PCAImage(img, face_C, face_U, eigen_vectors_L, col_picked_estimate1, 1);
	PCAImage(img, face_C, face_U, eigen_vectors_L, col_picked_estimate2, 2);
	PCAImage(img, face_C, face_U, eigen_vectors_L, col_picked_estimate3, 3);

	Mat eigen_vector_L = eigen_vectors_L.col(3);
	for (int i = 0; i < img_H; i++)
	{
		for (int j = 0; j < img_W; j++)
		{
			col_picked_estimate_L.at<float>(i, j) = eigen_vector_L.at<float>(i*img_W + j, 0);
		}
	}

}

void FaceRecognition::PCAImage(cv::Mat img, cv::Mat face_C, cv::Mat face_U, cv::Mat eigen_vectors_L, cv::Mat col_picked_estimate, int idx)
{
	using namespace cv;
	using namespace std;
	int H = face_C.rows;
	int W = face_C.cols;
	vector<double> coeffs_L(W, 0);
	Mat col_picked = face_U.col(idx);
	for (int i = 0; i < W; i++)
	{
		Mat eigen_vector = eigen_vectors_L.col(i);
		Mat coeff = eigen_vector.t()*col_picked;
		coeffs_L[i] = coeff.at<float>(0, 0);
	}
	int C_w = W;
	int C_h = H;
	H = img.rows;
	W = img.cols;
	Mat col_picked_estimate_ = Mat::zeros(Size(1, H*W), CV_32FC1);
	for (int i = 0; i < C_w; i++)
	{
		col_picked_estimate_ += coeffs_L[i] * eigen_vectors_L.col(i);
	}
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			col_picked_estimate.at<float>(i, j) = col_picked_estimate_.at<float>(i*W + j, 0);
		}
	}
}

void FaceRecognition::EigenFaces_training()
{
	using namespace cv;
	using namespace std;
	vector<Mat> imgs;
	vector<Mat> face_vectors;
	Mat face_U;
	vector<string> people_names;
	vector<int> people_names_numeral;

	int num =ReshapeImageAndGetName(people_names, people_names_numeral, face_U);

	//Mat face_L = face_U * face_U.t();
	Mat face_C = face_U.t() * face_U;
	int H = face_C.rows;
	int W = face_C.cols;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix(H, W);
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			matrix(i, j) = face_C.at<float>(i, j);
		}
	}
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(matrix);
	//cout << eig.eigenvalues() << endl;
	//cout << eig.eigenvectors() << endl;
	Mat eigen_values_C = Mat::zeros(Size(1, W), CV_32FC1);

	Mat eigen_vectors_C = Mat::zeros(Size(W, H), CV_32FC1);
	for (int i = 0; i < W; i++)
	{
		eigen_values_C.at<float>(i, 0) = eig.eigenvalues()(W - 1 - i, 0);

		Mat eigen_vector(Size(1, H), CV_32FC1);
		for (int j = 0; j < H; j++)
		{
			eigen_vector.at<float>(j, 0) = eig.eigenvectors().block(0, W - 1 - i, H, 1)(j, 0);
		}
		normalize(eigen_vector, eigen_vector);
		eigen_vector.copyTo(eigen_vectors_C.col(i));
	}

	Mat eigen_vectors_L = face_U * eigen_vectors_C;
	for (int i = 0; i < W; i++)
	{
		normalize(eigen_vectors_L.col(i), eigen_vectors_L.col(i));
	}
	vector<vector<double>> coeffs_L_all;
	vector<string> people_names_no_repetition;
	{
		int people_idx = 0;
		int accumulate = 0;
		for (int i = 0; i < num; i++)
		{
			static vector<double> coeffs_L_tmp(m_principleNum, 0);
			if (people_idx != people_names_numeral[i])
			{
				for (int j = 0; j < m_principleNum; j++)
				{
					coeffs_L_tmp[j] = coeffs_L_tmp[j] / accumulate;
				}
				coeffs_L_all.push_back(coeffs_L_tmp);
				coeffs_L_tmp.clear();
				people_names_no_repetition.push_back(people_names[i-1]);
				people_idx = people_names_numeral[i];
				accumulate = 0;
			}
			++accumulate;
			vector<double> coeffs_L = PCAImageCoeffs(face_C.cols, face_U, eigen_vectors_L, i);
			coeffs_L_tmp.resize(m_principleNum);
			for (int j = 0; j < m_principleNum; j++)
			{
				coeffs_L_tmp[j] += coeffs_L[j];
			}
		}
	}
#if 0
	{
		ofstream outfile;
		//ios::app
		//ios::trunc
		path = "F:/Datasets/LFW/eigen_space_coeffs.txt";
		outfile.open(path, ios::binary | ios::trunc | ios::in);
		H = face_C.rows;
		W = face_C.cols;
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				outfile << face_C.at<float>(i, j) << " ";
			}
			outfile << endl;
		}
		outfile.close();
	}
	{
		ofstream outfile;
		path = "F:/Datasets/LFW/face_C.txt";
		outfile.open(path, ios::binary | ios::trunc | ios::in);
		H = face_C.rows;
		W = face_C.cols;
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				outfile << face_C.at<float>(i, j) << " ";
			}
			outfile << endl;
		}
		outfile.close();

		path = "F:/Datasets/LFW/face_C_size.txt";
		outfile.open(path, ios::binary | ios::trunc | ios::in);
		outfile << H << " " << W << " ";
		outfile.close();
	}
	{
		ofstream outfile;
		path = "F:/Datasets/LFW/eigen_vectors_L.txt";
		outfile.open(path, ios::binary | ios::trunc | ios::in);
		H = eigen_vectors_L.rows;
		W = eigen_vectors_L.cols;
		for (int i = 0; i < H; i++)
		{
			for (int j = 0; j < W; j++)
			{
				outfile << eigen_vectors_L.at<float>(i, j) << " ";
			}
			outfile << endl;
		}
		outfile.close();

		path = "F:/Datasets/LFW/eigen_vectors_L_size.txt";
		outfile.open(path, ios::binary | ios::trunc | ios::in);
		outfile << H << " " << W << " ";
		outfile.close();
	}
#endif
	{
		stringstream ss;
		ss << num;
		string num_str = ss.str();
		string path = "F:/Datasets/LFW/eigen_faces_data_" + num_str + ".txt";
		cout << "write to file ... "<< path << endl;
		FileStorage fs(path, FileStorage::WRITE);
		fs << "face_C_cols" << face_C.cols;
		fs << "coeffs_L_all" << coeffs_L_all;
		fs << "people_names_numeral" << people_names_numeral;
		fs << "people_names" << people_names;
		fs << "people_names_no_repetition" << people_names_no_repetition;
		fs << "eigen_values_C" << eigen_values_C;
		fs << "eigen_vectors_L" << eigen_vectors_L.colRange(0, m_principleNum);
		fs.release();
		cout << "write done." << endl;
	}
}

void FaceRecognition::EigenFaces_test()
{
	using namespace cv;
	using namespace std;
	std::vector<std::string> img_list = TraverseFiles();
	int num = img_list.size();
	num = 100;
	vector<vector<double>> coeffs_L_all;
	vector<int> face_C_size;
	vector<int> eigen_vectors_L_size;
	vector<string> people_names;
	vector<int> people_names_numeral;
	vector<string> people_names_no_repetition;
#if 0
	{
		ifstream infile;
		string path = "F:/Datasets/LFW/face_C_size.txt";
		infile.open(path, ios::binary | ios::out);
		string tmp;
		while (getline(infile, tmp))
		{
			vector<double> coeffs_L;
			std::vector<std::string> segments;
			boost::split(segments, tmp, boost::is_any_of(" "));
			for (size_t i = 0; i < segments.size(); i++)
			{
				if (segments[i] != "")
				{
					stringstream ss;
					ss << segments[i];
					double coeff = 0.0;
					ss >> coeff;
					face_C_size.push_back(coeff);
				}
			}
		}
		infile.close();
	}
	{
		ifstream infile;
		string path = "F:/Datasets/LFW/eigen_vectors_L_size.txt";
		infile.open(path, ios::binary | ios::out);
		string tmp;
		while (getline(infile, tmp))
		{
			vector<double> coeffs_L;
			std::vector<std::string> segments;
			boost::split(segments, tmp, boost::is_any_of(" "));
			for (size_t i = 0; i < segments.size(); i++)
			{
				if (segments[i] != "")
				{
					stringstream ss;
					ss << segments[i];
					double coeff = 0.0;
					ss >> coeff;
					eigen_vectors_L_size.push_back(coeff);
				}
			}
		}
		infile.close();
	}

	Mat face_C = Mat::zeros(Size(face_C_size[1], face_C_size[0]), CV_32FC1);
	Mat eigen_vectors_L = Mat::zeros(Size(eigen_vectors_L_size[1], eigen_vectors_L_size[0]), CV_32FC1);

	{
		ifstream infile;
		string path = "F:/Datasets/LFW/eigen_space_coeffs.txt";
		infile.open(path, ios::binary | ios::out);
		string tmp;
		while (getline(infile, tmp))
		{
			vector<double> coeffs_L;
			std::vector<std::string> segments;
			boost::split(segments, tmp, boost::is_any_of(" "));
			for (size_t i = 0; i < segments.size(); i++)
			{
				if (segments[i] != "")
				{
					stringstream ss;
					ss << segments[i];
					double coeff = 0.0;
					ss >> coeff;
					coeffs_L.push_back(coeff);
				}
			}
			coeffs_L_all.push_back(coeffs_L);
		}
		infile.close();
	}
	{
		ifstream infile;
		string path = "F:/Datasets/LFW/face_C.txt";
		infile.open(path, ios::binary | ios::out);
		string tmp;
		int row_idx = 0;
		while (getline(infile, tmp))
		{
			vector<double> coeffs_L;
			std::vector<std::string> segments;
			boost::split(segments, tmp, boost::is_any_of(" "));
			for (size_t i = 0; i < segments.size(); i++)
			{
				if (segments[i] != "")
				{
					stringstream ss;
					ss << segments[i];
					double coeff = 0.0;
					ss >> coeff;
					face_C.at<float>(row_idx, i) = coeff;
				}
			}
			++row_idx;
		}
		infile.close();
	}
	{
		ifstream infile;
		string path = "F:/Datasets/LFW/eigen_vectors_L.txt";
		infile.open(path, ios::binary | ios::out);
		string tmp;
		int row_idx = 0;
		while (getline(infile, tmp))
		{
			vector<double> coeffs_L;
			std::vector<std::string> segments;
			boost::split(segments, tmp, boost::is_any_of(" "));
			for (size_t i = 0; i < segments.size(); i++)
			{
				if (segments[i] != "")
				{
					stringstream ss;
					ss << segments[i];
					double coeff = 0.0;
					ss >> coeff;
					eigen_vectors_L.at<float>(row_idx, i) = coeff;
				}
			}
			++row_idx;
		}
		infile.close();
	}

#endif
	int face_C_cols = 0;
	Mat eigen_vectors_L;
	Mat eigen_values_C;
	{
		stringstream ss;
		ss << num;
		string num_str = ss.str();
		string path = "F:/Datasets/LFW/eigen_faces_data_" + num_str + ".txt";
		cout << "restore from file ... " << path << endl;
		FileStorage fs(path, FileStorage::READ);
		fs["face_C_cols"] >> face_C_cols;
		fs["coeffs_L_all"] >> coeffs_L_all;
		fs ["people_names_numeral"] >> people_names_numeral;
		fs["people_names"] >> people_names;
		fs["people_names_no_repetition"]>> people_names_no_repetition;
		
		fs["eigen_values_C"] >> eigen_values_C;
		fs["eigen_vectors_L"] >> eigen_vectors_L;
		fs.release();
		cout << "restore from file done." << endl;
	}

	vector<Mat> imgs;
	vector<Mat> face_vectors;
	//string dir = "F:/Datasets/LFW/lfw-deepfunneled/Aaron_Tippin/";
	//string name = "Aaron_Tippin_0001.jpg";

	string dir = "F:/Datasets/LFW/other_people/trump/";
	string name = "1.jpg";
	string path = dir + name;
	Mat img;
	img = imread(path, 0);
	int lfw_imge_size = 250;
	resize(img,img,Size(lfw_imge_size, lfw_imge_size));
	int H = img.rows;
	int W = img.cols;
	Mat face_U_test(Size(1, H*W), CV_32FC1);
	for (int j = 0; j < H; j++)
	{
		for (int k = 0; k < W; k++)
		{
			face_U_test.at<float>(W*j + k, 0) = img.at<uchar>(j, k);
		}
	}
	vector<vector<double>> coeffs_L_all_test;
	vector<double> coeffs_L = PCAImageCoeffs(face_C_cols, face_U_test, eigen_vectors_L, 0);
	coeffs_L_all_test.push_back(coeffs_L);
	vector<double> lambdas;
	for (int i = 0; i < eigen_values_C.rows; i++)
	{
		lambdas.push_back(eigen_values_C.at<float>(i,0));
	}
	normalize(lambdas, lambdas);

	double smallest_err = 1e10;
	pair<int, int> match_pairs;
	cout << "matching ..." << endl;
	for (int j = 0; j < coeffs_L_all_test.size(); j++)
	{
		for (int i = 0; i < coeffs_L_all.size(); i++)
		{
			double err = norm(coeffs_L_all[i], coeffs_L_all_test[j]);
			//double err = MahalanobisDistance(coeffs_L_all[i], coeffs_L_all_test[j], lambdas);
			if (smallest_err > err)
			{
				smallest_err = err;
				match_pairs = make_pair(i, j);
			}
		}
	}
	cout << "matching finished" << endl;
	cout << "pairs:" << endl;

	cout << "pairs: " << match_pairs.first << ", " << match_pairs.second
		<< ". err: " << smallest_err << endl
		//<< "detect result: " << img_list[match_pairs.first]
		<< "detect result: " << people_names_no_repetition[match_pairs.first]
		<< endl;
	//img = imread(img_list[match_pairs.first], 0);

	path = img_list[0];
	int folder_idx_end = path.find_last_of("/");
	string path_folder = path.substr(0, folder_idx_end);
	int folder_idx_begin = path_folder.find_last_of("/");
	path_folder = path.substr(0, folder_idx_begin);
	path_folder += "/"+people_names_no_repetition[match_pairs.first] + "/" + people_names_no_repetition[match_pairs.first] + "_0001.jpg";
	img = imread(path_folder, 0);

	imshow("detected",img);
}

std::vector<std::string> FaceRecognition::TraverseFiles()
{
	using namespace cv;
	using namespace std;
	//string dir = "F:/Datasets/LFW/lfw";
	string dir = "F:/Datasets/LFW/lfw-deepfunneled";

	vector< string> fileList;
	find_file(const_cast<char*>(dir.c_str()), fileList);
	return fileList;
}

void FaceRecognition::find_file(char * lpPath, std::vector< std::string>& fileList)
{
	char find_file_name[MAX_PATH];
	WIN32_FIND_DATA FindFileData;

	strcpy(find_file_name, lpPath);
	strcat(find_file_name, "/*.*");

	HANDLE hFind = ::FindFirstFile(find_file_name, &FindFileData);
	if (INVALID_HANDLE_VALUE == hFind)
	{
		return;
	}
	while (true)
	{
		if (FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			if (FindFileData.cFileName[0] != '.')
			{
				char path[MAX_PATH];
				strcpy(path, lpPath);
				strcat(path, "/");
				strcat(path, (char*)(FindFileData.cFileName));
				find_file(path, fileList);
			}
		}
		else
		{
			char path[MAX_PATH];
			strcpy(path, lpPath);
			strcat(path, "/");
			strcat(path, (char*)(FindFileData.cFileName));
			fileList.push_back(path);
		}
		if (!FindNextFile(hFind, &FindFileData))
		{
			break;
		}
	}
	FindClose(hFind);
}

std::vector<double> FaceRecognition::PCAImageCoeffs(int face_C_W, cv::Mat face_U, cv::Mat eigen_vectors_L, int idx)
{
	using namespace cv;
	using namespace std;
	m_principleNum = face_C_W > 20 ? 20 : face_C_W;
	vector<double> coeffs_L(m_principleNum, 0);
	Mat col_picked = face_U.col(idx);

	for (int i = 0; i < m_principleNum; i++)
	{
		Mat eigen_vector = eigen_vectors_L.col(i);
		Mat coeff = eigen_vector.t()*col_picked;
		coeffs_L[i] = coeff.at<float>(0, 0);
	}
	return coeffs_L;
}

double FaceRecognition::MahalanobisDistance(std::vector<double> coeff, std::vector<double> coeff1, std::vector<double> lambdas)
{
	assert(coeff.size() == coeff1.size());
	double distance = 0.0;
	for (int i = 0; i < coeff.size(); i++)
	{
		distance += (coeff[i] - coeff1[i])*(coeff[i] - coeff1[i])/(lambdas[i]* lambdas[i]);
	}
	distance = sqrt(distance);
	return 0.0;
}

void FaceRecognition::SaveImageMatrixToFile()
{
	using namespace cv;
	using namespace std;
	Mat face_U;
	vector<string> people_names;
	vector<int> people_names_numeral;
	int num=ReshapeImageAndGetName(people_names, people_names_numeral, face_U);

	Mat people_names_numeral_mat(Size(1,people_names_numeral.size()),CV_16UC1);
	for (int i = 0; i < people_names_numeral.size(); i++)
	{
		people_names_numeral_mat.at<ushort>(i,0)= people_names_numeral[i];
	}

	stringstream ss;
	ss << num;
	string num_str=ss.str();

	string path = "F:/Datasets/LFW/raw_faces_data_" + num_str + ".txt";
	cout << "write to file ... " << path << endl;
	FileStorage fs(path, FileStorage::WRITE);
	fs << "people_names_numeral" << people_names_numeral;
	fs << "people_names_numeral_mat" << people_names_numeral_mat;
	fs << "people_names" << people_names;
	fs << "face_U" << face_U;
	fs.release();
	cout << "write done." << endl;
}

int FaceRecognition::ReshapeImageAndGetName( std::vector<std::string>& people_names, std::vector<int>& people_names_numeral,cv::Mat& face_U)
{
	using namespace cv;
	using namespace std;
	vector<string> img_list = TraverseFiles();
	int num = img_list.size();
	num = 100;
	string path = img_list[0];
	Mat img;
	img = imread(path, 0);
	int H = img.rows;
	int W = img.cols;
	face_U.create(Size(num, H*W), CV_32FC1);
	people_names_numeral.resize(num);
	cout << "reshape images ..." << endl;
	for (int i = 0; i < num; i++)
	{
		string path = img_list[i];
		int folder_idx_end = path.find_last_of("/");
		string path_folder = path.substr(0, folder_idx_end);
		int folder_idx_begin = path_folder.find_last_of("/");
		string people_name = path_folder.substr(folder_idx_begin + 1, folder_idx_end);
		people_names.push_back(people_name);

		Mat img = imread(path, 0);
		int H = img.rows;
		int W = img.cols;
		for (int j = 0; j < H; j++)
		{
			for (int k = 0; k < W; k++)
			{
				face_U.at<float>(W*j + k, i) = img.at<uchar>(j, k);
			}
		}
		if (i % 100 == 0)
		{
			cout << "reshape images " << i << "/" << num << endl;
		}
	}
	cout << "reshape images done" << endl;
	cout << "indexing images ..." << endl;

	{
		string tmp = "Aaron_Eckhart";
		int numeral = 0;
		for (int i = 0; i < num; i++)
		{
			if (tmp != people_names[i])
			{
				++numeral;
			}
			people_names_numeral[i] = numeral;
			tmp = people_names[i];
		}
	}
	cout << "indexing images done" << endl;
	return num;
}
