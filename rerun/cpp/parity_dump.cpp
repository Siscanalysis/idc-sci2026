// parity_dump.cpp — load an OpenNN .json network and dump measured-vs-predicted
// for every row of a CSV (no header; first n_in columns are inputs, the next
// column is the measured target). Uses the SAME model path the IDC driver uses,
// so the goodness-of-fit reflects the surrogate the optimizer actually queries
// (the exported NumPy .py predictor is unreliable for some holdout exports).
//
//   usage: parity_dump <nn_json> <data_csv> <n_in> <out_csv>

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include "standard_networks.h"
#include "bounding_layer.h"
#include "neural_network.h"

using namespace opennn;
using type = float;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    try
    {
        if(argc < 5) { std::cerr << "usage: " << argv[0] << " <nn_json> <data_csv> <n_in> <out_csv>\n"; return 2; }
        const fs::path nn_json = argv[1];
        const fs::path csv      = argv[2];
        const int      n_in     = std::stoi(argv[3]);
        const fs::path out_csv  = argv[4];

        std::vector<std::vector<type>> rows;
        std::ifstream f(csv); std::string line;
        while(std::getline(f, line))
        {
            if(line.empty()) continue;
            std::vector<type> r; std::stringstream ss(line); std::string cell;
            while(std::getline(ss, cell, ',')) { try { r.push_back((type)std::stod(cell)); } catch(...) { r.clear(); break; } }
            if((int)r.size() >= n_in + 1) rows.push_back(r);
        }
        const int n = (int)rows.size();
        if(n == 0) throw std::runtime_error("no numeric rows parsed from CSV");

        ApproximationNetwork net({Index(n_in)}, Shape{Index(8)}, {Index(1)});
        if(auto* b = dynamic_cast<Bounding*>(net.get_first("Bounding"))) b->set_bounding_method("NoBounding");
        if(!fs::exists(nn_json)) throw std::runtime_error("NN file missing: " + nn_json.string());
        net.load(nn_json.string());
        if(auto* b = dynamic_cast<Bounding*>(net.get_first("Bounding"))) b->set_bounding_method("NoBounding");
        {
            auto vi = net.get_input_variables();  for(auto& v : vi) v.set_role("Input");  net.set_input_variables(vi);
            auto vo = net.get_output_variables(); for(auto& v : vo) v.set_role("Target"); net.set_output_variables(vo);
        }

        MatrixR X(n, n_in);
        for(int i = 0; i < n; ++i) for(int j = 0; j < n_in; ++j) X(i, j) = rows[i][j];
        MatrixR Y = net.calculate_outputs(X);

        std::ofstream o(out_csv); o << "measured,predicted\n" << std::setprecision(9);
        for(int i = 0; i < n; ++i) o << rows[i][n_in] << "," << Y(i, 0) << "\n";
        std::cout << "[OK] " << n << " rows -> " << out_csv << std::endl;
        return 0;
    }
    catch(const std::exception& e) { std::cerr << "[ERROR] " << e.what() << std::endl; return 1; }
}
