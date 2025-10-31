#if 1
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bitset>
#include <array>
#include <fstream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <set>

namespace py = pybind11;

// Per dataset v1
//std::unordered_map<int, std::string> gestureMap = {
//		{1, "[YEAH]"},
//		{2, "[Full Hand]"},
//		{3, "[3 Fingers]"},
//		{4, "[2 Fingers]"},
//		{0, "[Noise or Unmapped]"}
//};
std::unordered_map<int, std::string> gestureMap = {
		{1, "[4 Fingers]"},
		{2, "[4 Fingers L]"},
		{3, "[Scorn]"},
		{4, "[Open Hand]"},
		{5, "[Closed Hand]"},
		{6, "[2 Fingers]"},
		{7, "[2 Fingers L]"},
		{8, "[3 Fingers]"},
		{9, "[YEAH]"},
		{0, "[Noise or Unmapped]"} // Non presente nel dataset 3, 4
};

std::pair<int, double> predictFromData(double* data, const py::object& model, const py::object& scaler, const py::object& pd, const py::list& columns) {
	// Crea l'array numpy
	size_t shape[2] = { 1, 24 };
	py::array_t<double> input_array(shape, data);

	// Converte in DataFrame con colonne corrette
	py::object input_df = pd.attr("DataFrame")(input_array, py::arg("columns") = columns);

	// Normalizza con scaler
	py::object input_normalized = scaler.attr("transform")(input_df);

	// Ottieni le probabilità (shape: [1, n_classes])
	py::object probabilities = model.attr("predict_proba")(input_normalized);

	// Prima (e unica) riga delle probabilità
	py::object prob_row = probabilities.attr("__getitem__")(0);

	// Trova indice della probabilità massima (argmax)
	int predicted_index = prob_row.attr("argmax")().cast<int>();

	// Recupera la lista delle classi dal modello
	py::list classes = model.attr("classes_");

	// Mappa l'indice al valore della classe reale
	int predicted_class = classes[predicted_index].cast<int>();

	// Recupera la confidenza (valore massimo)
	double confidence = prob_row.attr("max")().cast<double>();

	return { predicted_class, confidence };
}


void removeSpecificIndices(std::vector<double>& vec) {
	// Indici da rimuovere
	std::set<size_t> indicesToRemove = { 0,1,2,3,25,27,29,31,32 };

	// Rimuove partendo dal più grande per evitare shift di indici
	for (auto it = indicesToRemove.rbegin(); it != indicesToRemove.rend(); ++it) {
		if (*it < vec.size()) {
			vec.erase(vec.begin() + *it);
		}
	}
}

std::vector<double> splitStringToDoubleVector(const std::string& input) {
	std::vector<double> numbers;
	std::stringstream ss(input);
	std::string item;

	while (std::getline(ss, item, ',')) {
		numbers.push_back(std::stod(item));
	}

	return numbers;
}

// Mappa il minAreaRect ad un rettangolo 100x400
cv::Mat PerspectiveTransform(const cv::Mat& img, const cv::Point2f* src_points, const cv::Size& target_size)
{
	cv::Point2f dst_points[4]{
		cv::Point2f(0, 0),
		cv::Point2f(target_size.width - 1, 0),
		cv::Point2f(target_size.width - 1, target_size.height - 1),
		cv::Point2f(0, target_size.height - 1)
	};

	const cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

	cv::Mat warped_img;
	cv::warpPerspective(img, warped_img, perspective_matrix, target_size);

	return warped_img;
}

// Funzione per ordinare i punti in senso orario (top-left, top-right, bottom-right, bottom-left)
void OrderPoints(const cv::Point2f* points, cv::Point2f* dest, size_t array_size)
{

	cv::Point2f center(0, 0);

	for (int i = 0; i < array_size; i++)
	{
		const cv::Point2f point = points[i];
		dest[i] = point;
		center += point;
	}
	center *= (1.0 / array_size);

	// Ordina i punti in base all'angolo rispetto al centro
	std::sort(dest, dest + array_size, [&](const cv::Point2f& a, const cv::Point2f& b) {
		return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
		});
}

// Binarizzazione dei marker
//  - bits [0..3] = sections 0..2 (1 = color1, 0 = color2)
//  - bits [4..7] = error flags 
std::bitset<8> Roi2Bin(cv::Mat& roi)
{
	cv::GaussianBlur(roi, roi, cv::Size(5, 5), 0); // opzionale ma aiuta con l'HSV
	cv::cvtColor(roi, roi, cv::COLOR_BGR2HSV);

	const int height = roi.rows;
	const int width = roi.cols;

	std::bitset<8> marker_bin{};

	// Dividi l'immagine in 4 sezioni orizzontali
	constexpr unsigned num_sections = 4;
	const cv::Mat sections[num_sections]
	{
		roi(cv::Rect(cv::Point(0, 0), cv::Point(width, height / 4))),
		roi(cv::Rect(cv::Point(0, height / 4), cv::Point(width, 2 * height / 4))),
		roi(cv::Rect(cv::Point(0, 2 * height / 4), cv::Point(width, 3 * height / 4))),
		roi(cv::Rect(cv::Point(0, 3 * height / 4), cv::Point(width, height)))
	};

	constexpr int color1maxHue = 179, color1minHue = 120;

	constexpr int color1maxHue2 = 15, color1minHue2 = 0;
	constexpr int color2maxHue = 60, color2minHue = 16;

	// Trova il colore dominante (hue) in ogni sezione tramite l'istogramma

	constexpr int hist_size = 180;
	constexpr float range[] = { 0.f, 180.f };
	const float* histRange[] = { range };

	for (unsigned i = 0; i < num_sections; i++)
	{
		const cv::Mat& section = sections[i];

		cv::Mat hist;
		int channels[]{ 0 }; // Hue
		cv::calcHist(&section, 1, channels, cv::Mat(), hist, 1, &hist_size, histRange);

		cv::Point max_loc;
		cv::minMaxLoc(hist, nullptr, nullptr, nullptr, &max_loc);

		std::cout << "Hue Dominante nella sezione: " << max_loc.y << std::endl;

		if ((max_loc.y >= color1minHue && max_loc.y <= color1maxHue) ||
			(max_loc.y >= color1minHue2 && max_loc.y <= color1maxHue2))
		{
			marker_bin.set(i, true);
		}
		else if (max_loc.y >= color2minHue && max_loc.y <= color2maxHue)
		{
			marker_bin.set(i, false);
		}
		else // Errore
		{
			marker_bin.set(i + 4, true);
		}
	}

	return marker_bin;
}

namespace proj
{
	struct Point
	{
		Point(cv::Point p, cv::Point origin, std::bitset<4> t_color_bits)
		{
			point = p - origin;
			color_bits = t_color_bits;
		}

		cv::Point point;
		std::bitset<4> color_bits;
	};

	void SortPointsByColorBits(std::vector<Point>& points)
	{
		// Ordine desiderato
		std::vector<std::string> desired_order = { "0000", "1111", "1100", "0011", "1010", "0101" };

		// Mappa da stringa di bit a posizione
		std::unordered_map<std::string, int> order_map;
		for (size_t i = 0; i < desired_order.size(); ++i)
		{
			order_map[desired_order[i]] = static_cast<int>(i);
		}

		// Ordina il vettore usando il map come criterio (bisogna controllare prima di questo che non ci siano ID duplicati, in quel caso la lettura viene scartata)
		std::sort(points.begin(), points.end(), [&](const Point& a, const Point& b) {
			return order_map[a.color_bits.to_string()] < order_map[b.color_bits.to_string()];
			});
	}

	class Frame
	{
	public:

		std::vector<Point> markers_in_frame;
		int frame_id;

		// mappa ID -> coordinate relative (popolata in constructor)
		std::map<std::string, cv::Point> relative_coords;

		// distanze per coppie specificate
		std::map<std::pair<std::string, std::string>, double> pair_distances;

		// coppie di cui vogliamo la distanza
		static const std::vector<std::pair<std::string, std::string>> DISTANCE_PAIRS;

		Frame(std::vector<Point> frame_points, int id)
			: markers_in_frame(std::move(frame_points)), frame_id(id)
		{
			SortPointsByColorBits(markers_in_frame);
			computeRelativeCoordsAndDistances();
			printGeometricInfo();
		}

	private:

		void computeRelativeCoordsAndDistances()
		{
			// 1) mappa ID -> punto assoluto
			std::map<std::string, cv::Point> abs_pts;
			for (auto const& pt : markers_in_frame) {
				abs_pts[pt.color_bits.to_string()] = pt.point;
			}

			// riferimento "0000"
			const cv::Point ref = abs_pts.at("0000");

			// 2) coordinate relative
			for (auto const& [id, p] : abs_pts) {
				relative_coords[id] = p - ref;
			}

			// 3) distanze per coppie
			for (auto const& pr : DISTANCE_PAIRS) {
				const auto& id1 = pr.first;
				const auto& id2 = pr.second;
				const cv::Point& p1 = abs_pts.at(id1);
				const cv::Point& p2 = abs_pts.at(id2);
				double d = std::hypot(p2.x - p1.x, p2.y - p1.y);
				pair_distances[{id1, id2}] = d;
			}
		}

	public:

		void printGeometricInfo() const
		{
			std::cout << "[Frame " << frame_id << "]\n";
			std::cout << "Coordinate relative al riferimento (0000):\n";
			for (auto const& [id, rel] : relative_coords) {
				std::cout << "  ID " << id
					<< ": (" << rel.x << ", " << rel.y << ")\n";
			}

			std::cout << "Distanze euclidee:\n";
			for (auto const& pr : DISTANCE_PAIRS) {
				double d = pair_distances.at(pr);
				std::cout << "  d(" << pr.first << "," << pr.second
					<< ") = " << d << "\n";
			}
			std::cout << "---------------------------------------\n";
		}
	};

	const std::vector<std::pair<std::string, std::string>> Frame::DISTANCE_PAIRS = {
		{"0101","1100"},
		{"1100","0101"},
		{"1100","1010"},
		{"1010","1100"},
		{"1010","0011"},
		{"0011","1010"},
		{"0011","1111"},
		{"1111","0011"}
	};


	void FillMissingColorBits(std::vector<Point>& points, cv::Point origin = cv::Point(0, 0))
	{
		// ID da controllare
		std::vector<std::bitset<4>> all_ids = {
			std::bitset<4>("0000"),
			std::bitset<4>("1111"),
			std::bitset<4>("1100"),
			std::bitset<4>("0011"),
			std::bitset<4>("1010"),
			std::bitset<4>("0101")
		};

		// Salva gli ID dei marker presenti nel frame usando la versione intera
		std::set<unsigned long> existing_ids;
		for (const auto& pt : points)
		{
			existing_ids.insert(pt.color_bits.to_ulong());
		}

		// Aggiungi quelli mancanti con coordinate 0,0
		for (const auto& id : all_ids)
		{
			if (existing_ids.find(id.to_ulong()) == existing_ids.end())
			{
				points.emplace_back(cv::Point(0, 0), origin, id);
			}
		}
	}

	static const std::vector<std::string> CSV_IDS = { "0000","1111","1100","0011","1010","0101" };
	static const std::vector<std::pair<std::string, std::string>> CSV_PAIRS = {
		{"0101","1100"},{"1100","0101"},{"1100","1010"},{"1010","1100"},
		{"1010","0011"},{"0011","1010"},{"0011","1111"},{"1111","0011"}
	};

	// Generazione header CSV
	std::string GenerateCsvHeader()
	{
		std::ostringstream hdr;
		for (auto& id : CSV_IDS) {
			hdr << "coord_" << id << "_X,coord_" << id << "_Y,origin_dist_" << id << ",is_visible_" << id << ",";
		}
		for (auto& pr : CSV_PAIRS)
			hdr << "d_" << pr.first << "_" << pr.second << ",";
		hdr << "reading_id\n";
		return hdr.str();
	}

	// Generazione riga CSV per frame
	std::string GenerateCsvLine(
		const std::vector<Point>& points,
		const std::set<std::string>& visible_ids,
		int reading_id)
	{
		std::map<std::string, cv::Point> mp;
		for (auto& pt : points)
			mp[pt.color_bits.to_string()] = pt.point;

		std::ostringstream line;
		for (auto& id : CSV_IDS)
		{
			auto p = mp[id];
			line << p.x << "," << p.y << "," << std::hypot(p.x, p.y) << "," << (visible_ids.count(id) ? 1 : 0) << ",";
		}
		for (auto& pr : CSV_PAIRS)
		{
			auto p1 = mp[pr.first], p2 = mp[pr.second];
			line << std::hypot(p2.x - p1.x, p2.y - p1.y) << ",";
		}
		line << reading_id << "\n";
		return line.str();
	}

	bool ValidateColorBits(const std::vector<proj::Point>& points)
	{
		// Lista di ID validi
		std::set<std::string> valid_ids = {
			"0000", "1111", "1100", "0011", "1010", "0101"
		};

		std::set<std::string> seen_ids;
		bool found_0000 = false;

		//std::cout << "FOUND IDs IN FRAME: " << std::endl;
		for (const auto& pt : points)
		{
			std::string id_str = pt.color_bits.to_string();
			//std::cout << id_str << std::endl;

			// Controlla se l'ID è valido
			if (valid_ids.find(id_str) == valid_ids.end())
			{
				return false; // ID non valido
			}

			// Controlla se l'ID è duplicato
			if (!seen_ids.insert(id_str).second)
			{
				return false; // ID duplicato
			}

			// Controlla se abbiamo visto l'ID "0000"
			if (id_str == "0000")
			{
				found_0000 = true;
			}
		}

		// Verifica che "0000" sia presente
		if (!found_0000)
		{
			return false;
		}

		return true;
	}



}

int main()
{
	py::scoped_interpreter guard{};

	// === FASE PRELIMINARE ML ===
	py::module_ joblib = py::module_::import("joblib");
	py::module_ pd = py::module_::import("pandas");

	std::cout << "Caricamento scaler e modello..." << std::endl;

	// Carica scaler e modello
	py::object scaler = joblib.attr("load")("pybindSave/scaler_new_proba.joblib"); // _old per Dataset 1
	py::object model = joblib.attr("load")("pybindSave/svm_model_new_proba.joblib");

	std::cout << "Caricamento colonne dal dataset..." << std::endl;

	// Carica la lista dei nomi colonne dal file originale usato nel training
	py::object pd_read = pd.attr("read_csv");
	py::object train_df = pd_read("datasets/gesture_train_dataset_4.csv"); // _v1 per Dataset 1

	// Rimuovi colonne droppate + target
	py::list drop_columns;
	drop_columns.append("coord_0000_X");
	drop_columns.append("coord_0000_Y");
	drop_columns.append("origin_dist_0000");
	drop_columns.append("is_visible_0000");
	drop_columns.append("d_1100_0101");
	drop_columns.append("d_1010_1100");
	drop_columns.append("d_0011_1010");
	drop_columns.append("d_1111_0011");
	train_df = train_df.attr("drop")(drop_columns, py::arg("axis") = 1);

	train_df = train_df.attr("drop")("reading_id", py::arg("axis") = 1);
	py::list columns = train_df.attr("columns");

	std::cout << "Preset preliminare completato." << std::endl;

	// === FASE IPA ===
	//constexpr const char* Video_File_Path{ R"(C:\Users\Cipher\Desktop\Universitade Mk2\Image Analysis\VSC\OpenCVexc\src\example_images\hando10.mp4)" };
	//constexpr const char* Video_File_Path{ R"(videos\handoCompTest8.mp4)" }; // handoCompTest6.mp4 complessivo Dataset 1

	std::string filename;
	const std::string folder = "videos\\";
	std::cout << "Inserisci il nome del file video (es. handoCompTest8.mp4): ";
	std::getline(std::cin, filename);
	std::string Video_File_Path = folder + filename;

	std::cout << "Percorso completo:" << Video_File_Path << std::endl;
	cv::VideoCapture cap(Video_File_Path);

	const double fps = cap.get(cv::CAP_PROP_FPS);
	const int delay = (int)(1000 / fps);
	if (!cap.isOpened())
	{
		std::cerr << "Errore: impossibile aprire il file video " << Video_File_Path << std::endl;
		return -1;
	}

	cv::Mat frame;
	cv::namedWindow("Frame", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Glove Mask", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("All Warped ROIs (Horizontal)", cv::WINDOW_AUTOSIZE);

	std::vector<std::vector<cv::Point>> contours;
	const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::Mat color1, color2_1, color2_2, mask, frameHSV;

	std::streambuf* orig_buf = std::cout.rdbuf();
	std::ostream null_stream(nullptr);
	std::cout.rdbuf(null_stream.rdbuf());
	std::streambuf* orig_cerr = std::cerr.rdbuf();
	std::cerr.rdbuf(null_stream.rdbuf());


	const auto resized = [](const cv::Mat& src, cv::Size size) -> cv::Mat {
		cv::Mat dst;
		cv::resize(src, dst, size);
		return dst;
		};

	constexpr double min_contour_area = 60.0;

	std::vector<proj::Point> points;
	std::vector<proj::Frame> frames;
	int framecount = 0;
	int gesture_id = 9;

	//std::ofstream csv_out(R"(C:\Users\Cipher\Desktop\Universitade Mk2\Image Analysis\VSC\OpenCVexc\src\example_images\output.csv)");
	std::ofstream csv_out(R"(csv\output.csv)");
	csv_out << proj::GenerateCsvHeader();

	int droppedCount = 0;
	while (cap.read(frame))
	{
		cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
		contours.clear();
		points.clear();
		framecount++;

		cv::GaussianBlur(frame, frame, cv::Size(5, 5), 5);
		frameHSV = frame.clone();
		cv::resize(frame, frame, cv::Size(frame.cols * 0.5, frame.rows * 0.5));
		cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
		cv::inRange(frameHSV, cv::Scalar(20, 100, 100), cv::Scalar(35, 255, 255), color1);
		cv::inRange(frameHSV, cv::Scalar(120, 100, 100), cv::Scalar(179, 255, 255), color2_1);
		cv::inRange(frameHSV, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), color2_2);
		cv::bitwise_or(color2_1, color2_2, mask);
		cv::bitwise_or(color1, mask, mask);
		cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		std::vector<cv::Mat> all_warped_rois;
		cv::Mat frame_copy = frame.clone(); // Copia del frame per l'analisi della ROI

		for (const auto& contour : contours)
		{
			const double area = cv::contourArea(contour);
			if (area > min_contour_area) {
				cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
				cv::Point2f rect_points_float[4];
				rotated_rect.points(rect_points_float);

				// ROI dalla copia del frame per non dar fastidio alla lettura dei marker (perchè su frame ho messo tutti i contorni, scritte ecc...)
				cv::Rect bounding_rect = rotated_rect.boundingRect(); // VECCHIO, RAW ROI NON PIU USATO, SOLO TESTING
				bounding_rect &= cv::Rect(0, 0, frame_copy.cols, frame_copy.rows);
				cv::Mat raw_roi = frame_copy(bounding_rect).clone();

				cv::Point2f src_points_roi[4]{};
				OrderPoints(rect_points_float, src_points_roi, 4);
				cv::Mat warped_raw_roi = PerspectiveTransform(frame, src_points_roi, cv::Size(10, 40));
				//cv::imshow("raw roi", warped_raw_roi);
				//cv::imshow("raw roi2", raw_roi);

				// Leggi dalla ROI il codice binario del marker
				const std::bitset<8> binary_code = Roi2Bin(warped_raw_roi);
				std::cout << "Binary code: " << binary_code << '\n';

				std::bitset<4> error_flags;
				for (int i = 0; i < 4; ++i)
					error_flags.set(i, binary_code.test(i + 4));

				std::cout << "Error code: " << error_flags << '\n';

				std::bitset<4> color_bits;
				for (int i = 0; i < 4; ++i)
					color_bits.set(i, binary_code.test(i));

				std::cout << "Color bits: " << color_bits << '\n';

				// Calcola il centro del minAreaRect
				const cv::Point2f center = rotated_rect.center;
				const cv::Point center_int(static_cast<int>(center.x), static_cast<int>(center.y));

				// Stampa il centro sulla console
				std::cout << "ID: " << binary_code << ", Position: (" << center_int.x << ", " << center_int.y << ")" << std::endl;

				// es. marker di riferimento centrale, poi dovra essere rilevato dinamicamente il marker del polso
				cv::Point ref_marker(frame.cols / 2, frame.rows / 2);
				points.emplace_back(center_int, ref_marker, color_bits); // colleziona punto
				std::cout << "Point (relativo): " << points.back().point << '\n';
				std::cout << "===========================================\n";

				// disegno origine
				{
					cv::circle(frame, ref_marker, 5, cv::Scalar(0, 0, 255), 2);
					cv::drawMarker(
						frame,
						ref_marker,
						cv::Scalar(0, 255, 0),
						cv::MARKER_CROSS,
						20,
						2
					);
				}

				// Applica la trasformazione prospettica (per la visualizzazione)
				cv::Point2f src_points[4]{};
				OrderPoints(rect_points_float, src_points, 4);

				const cv::Mat warped_roi = PerspectiveTransform(frame, src_points, cv::Size(100, 400));
				all_warped_rois.push_back(warped_roi);

				// Disegna il centro del minAreaRect
				cv::circle(frame, center_int, 1, cv::Scalar(255, 0, 0), -1);

				// Disegna il minAreaRect sul frame originale
				constexpr unsigned num_draw_points = 4;
				cv::Point draw_points[num_draw_points]{};
				for (int i = 0; i < num_draw_points; ++i)
				{
					draw_points[i] = rect_points_float[i];
				}
				const cv::Point* const pts = draw_points;
				const int npts = 4;
				cv::polylines(frame, &pts, &npts, 1, true, cv::Scalar(0, 0, 255), 2);

				// Calcola la posizione per il testo (sopra il rettangolo)
				const cv::Point text_position(
					static_cast<int>(rotated_rect.center.x - 10),
					static_cast<int>(rotated_rect.center.y - rotated_rect.size.height / 2 - 10)
				);

				if (error_flags.any())
				{
					cv::putText(frame,
						"READING ERROR: " + color_bits.to_string() + " (e:" + error_flags.to_string() + ")",
						text_position,
						cv::FONT_HERSHEY_SIMPLEX,
						0.4,
						cv::Scalar(0, 0, 255), 2
					);
				}
				else
				{
					cv::putText(frame,
						color_bits.to_string(),
						text_position,
						cv::FONT_HERSHEY_SIMPLEX,
						0.4,
						cv::Scalar(0, 255, 0), 2
					);
					// MACHINE LEARNING QUI CREDO
				}
			}
		}

		if (const int num_rois = all_warped_rois.size())
		{
			const cv::Mat combined_roi(400, num_rois * 100, CV_8UC3, cv::Scalar(0));

			for (int i = 0; i < num_rois; ++i)
			{
				cv::Mat roi = all_warped_rois[i];
				if (i * 100 < combined_roi.cols)
				{
					roi.copyTo(combined_roi(cv::Rect(i * 100, 0, std::min(100, combined_roi.cols - i * 100), 400)));
				}
			}
			cv::imshow("All Warped ROIs (Horizontal)", combined_roi);
		}
		else
		{
			cv::Mat empty_roi(400, 100, CV_8UC3, cv::Scalar(0));
			cv::imshow("All Warped ROIs (Horizontal)", empty_roi);
		}

		std::set<std::string> vis;
		for (auto& pt : points) vis.insert(pt.color_bits.to_string());
		if (!ValidateColorBits(points))
		{
			std::cerr << "Errore: ID non validi o duplicati rilevati." << std::endl;
			cv::putText(frame,
				"[FRAME DROPPED]",
				cv::Point(20, 20),
				cv::FONT_HERSHEY_SIMPLEX,
				0.4,
				cv::Scalar(0, 0, 255), 2
			);
			droppedCount++;
		}
		else
		{
			// tutte le letture OK, da qui si possono iniziare a fare considerazioni geometriche
			proj::FillMissingColorBits(points);
			frames.emplace_back(points, framecount);
			std::string csvLine = proj::GenerateCsvLine(points, vis, gesture_id);
			csv_out << csvLine;
			std::vector<double> data = splitStringToDoubleVector(csvLine);
			removeSpecificIndices(data);
			auto [class_prediction, confidence] = predictFromData(data.data(), model, scaler, pd, columns);
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << confidence * 100;
			std::string conf_str = ss.str();
			cv::Scalar text_color = (confidence <= 0.75) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);

			if (gestureMap.find(class_prediction) != gestureMap.end()) {
				cv::putText(frame,
					gestureMap[class_prediction] + " conf: " + conf_str + "%",
					cv::Point(20, 20),
					cv::FONT_HERSHEY_SIMPLEX,
					0.6,
					text_color, 2
				);
			}
			else {
				cv::putText(frame,
					"[PRED: " + std::to_string(class_prediction) + "] conf: " + conf_str + "%",
					cv::Point(20, 20),
					cv::FONT_HERSHEY_SIMPLEX,
					0.6,
					text_color, 2
				);
			}
		}

		cv::imshow("Glove Mask", mask);
		cv::imshow("Frame", frame);
		cv::waitKey(delay);// "delay" per real time, "0" per frame by frame
	}

	cap.release();
	cv::destroyAllWindows();

	std::cout.rdbuf(orig_buf); // riattiva cout
	std::cerr.rdbuf(orig_cerr);

	for (const auto& f : frames)
	{
		f.printGeometricInfo();
	}


	for (auto pointset : frames)
	{
		std::cout << "[-----------" << pointset.frame_id << "--------------]" << std::endl;
		for (auto point : pointset.markers_in_frame)
		{
			std::cout << "ID: " << point.color_bits << " coords: " << point.point << std::endl;
		}
	}
	std::cout << "DROPPED COUNT: " << droppedCount << std::endl;
	std::cout << "FRAME COUNT: " << framecount << std::endl;
	float percentage = static_cast<float>(droppedCount) / framecount * 100.0f;

	std::cout << "DROPPED " << std::fixed << std::setprecision(2) << percentage << "% OF FRAMES" << std::endl;
	csv_out.close();

	return 0;
}
#endif