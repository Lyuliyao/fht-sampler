/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "./bias/Bias.h"
#include "./bias/ActionRegister.h"
#include "./core/PlumedMain.h"
#include "./core/Atoms.h"
#include "./core/FlexibleBin.h"
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <map>
#include <cmath>
#include <stdexcept>  // Add for std::runtime_error
#include <algorithm>
#include <limits>
namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS METATENSOR
/*
Adds harmonic and/or linear metatensors on one or more variables.

Either or both
of SLOPE and KAPPA must be present to specify the linear and harmonic force constants
respectively.  The resulting potential is given by:
\f[
  \sum_i \frac{k_i}{2} (x_i-a_i)^2 + m_i*(x_i-a_i)
\f].

The number of components for any vector of force constants must be equal to the number
of arguments to the action.

Additional material and examples can be also found in the tutorial \ref lugano-2

\par Examples

The following input tells plumed to restrain the distance between atoms 3 and 5
and the distance between atoms 2 and 4, at different equilibrium
values, and to print the energy of the metatensor
\plumedfile
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
METATENSOR ARG=d1,d2 AT=1.0,1.5 KAPPA=150.0,150.0 LABEL=metatensor
PRINT ARG=metatensor.bias
\endplumedfile

*/
//+ENDPLUMEDOC

struct TensorData {
  std::vector<uint32_t> shape;
  std::vector<std::vector<double>> data2D;          // For 2D tensors (e.g., (31, 4))
  std::vector<std::vector<std::vector<double>>> data3D; // For 3D tensors (e.g., (2, 4, 4))
};

static std::string trim_copy(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return "";
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

static bool load_basis_params_file(const std::string& filename, double& sigma_out, double& center_spacing_out) {
  std::ifstream in(filename);
  if (!in.is_open()) return false;

  bool has_sigma = false;
  bool has_center_spacing = false;
  std::string line;
  while (std::getline(in, line)) {
    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
    line = trim_copy(line);
    if (line.empty()) continue;

    const auto eq = line.find('=');
    if (eq == std::string::npos) continue;

    const std::string key = trim_copy(line.substr(0, eq));
    const std::string val = trim_copy(line.substr(eq + 1));
    if (key.empty() || val.empty()) continue;

    double parsed = 0.0;
    try {
      parsed = std::stod(val);
    } catch (...) {
      continue;
    }

    if (key == "sigma") {
      sigma_out = parsed;
      has_sigma = true;
    } else if (key == "center_spacing") {
      center_spacing_out = parsed;
      has_center_spacing = true;
    }
  }
  return has_sigma && has_center_spacing;
}

static bool load_basis_params_from_dirs(
    const std::vector<std::string>& dirs,
    double& sigma_out,
    double& center_spacing_out,
    std::string& source_file) {
  for (const auto& dir : dirs) {
    if (dir.empty()) continue;
    const std::string filename = dir + "/basis_params.dat";
    double s_tmp = sigma_out;
    double cs_tmp = center_spacing_out;
    if (load_basis_params_file(filename, s_tmp, cs_tmp)) {
      sigma_out = s_tmp;
      center_spacing_out = cs_tmp;
      source_file = filename;
      return true;
    }
  }
  return false;
}


class TensorCollection {
  private:
      std::map<std::string, TensorData> tensors;
  
      // Reshape flat data into 2D or 3D based on shape
      void reshape_data(TensorData& tensor, const std::vector<double>& flat_data) {
          if (tensor.shape.size() == 2) {
              // Reshape into 2D
              size_t rows = tensor.shape[0];
              size_t cols = tensor.shape[1];
              tensor.data2D.resize(rows);
              for (size_t i = 0; i < rows; ++i) {
                  tensor.data2D[i].resize(cols);
                  for (size_t j = 0; j < cols; ++j) {
                      tensor.data2D[i][j] = flat_data[i * cols + j];
                  }
              }
          } else if (tensor.shape.size() == 3) {
              // Reshape into 3D
              size_t depth = tensor.shape[0];
              size_t rows = tensor.shape[1];
              size_t cols = tensor.shape[2];
              tensor.data3D.resize(depth);
              for (size_t d = 0; d < depth; ++d) {
                  tensor.data3D[d].resize(rows);
                  for (size_t i = 0; i < rows; ++i) {
                      tensor.data3D[d][i].resize(cols);
                      for (size_t j = 0; j < cols; ++j) {
                          size_t idx = d * rows * cols + i * cols + j;
                          tensor.data3D[d][i][j] = flat_data[idx];
                      }
                  }
              }
          } else {
              throw std::runtime_error("Unsupported tensor dimension");
          }
      }
  
  public:
      // Load a tensor from file and reshape it
      void update(const std::string& key_str, const TensorData& tensor) {
          tensors[key_str] = tensor;
      }
      void load(const std::string& key_str, const std::string& filename) {
          TensorData tensor;
          std::ifstream file(filename, std::ios::binary);
          if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);
  
          // Read key metadata (skip key for simplicity)
          uint32_t key_len;
          file.read(reinterpret_cast<char*>(&key_len), sizeof(uint32_t));
          file.ignore(key_len);  // Skip key string
  
          // Read shape
          uint32_t ndims;
          file.read(reinterpret_cast<char*>(&ndims), sizeof(uint32_t));
          tensor.shape.resize(ndims);
          file.read(reinterpret_cast<char*>(tensor.shape.data()), ndims * sizeof(uint32_t));
  
          // Read flat data
          size_t num_elements = 1;
          for (auto dim : tensor.shape) num_elements *= dim;
          std::vector<double> flat_data(num_elements);
          file.read(reinterpret_cast<char*>(flat_data.data()), num_elements * sizeof(double));
  
          // Reshape into 2D/3D
          reshape_data(tensor, flat_data);
  
          tensors[key_str] = tensor;
      }
  
      // Access 2D tensor
      const std::vector<std::vector<double>>& get2D(const std::string& key) const {
          auto it = tensors.find(key);
          if (it == tensors.end() || it->second.shape.size() != 2) {
              throw std::runtime_error("2D tensor not found: " + key);
          }
          return it->second.data2D;
      }
  
      // Access 3D tensor
      const std::vector<std::vector<std::vector<double>>>& get3D(const std::string& key) const {
          auto it = tensors.find(key);
          if (it == tensors.end() || it->second.shape.size() != 3) {
              throw std::runtime_error("3D tensor not found: " + key);
          }
          return it->second.data3D;
      }
  };


class Rho {
  private:
  TensorCollection collection;
  int depth;
  int deg;
  int d_true;
  int d_pad;
  int n_modes;
  double sigma;
  double center_spacing;
  bool pbc;
  double period;
  double domain_min;
  double domain_max;
  bool whiten;
  double whitening_regularization;
  bool normalize;
  std::vector<std::vector<double>> whitening_matrix;
  std::vector<double> basis_integrals;
  double normalization_constant;

  static std::vector<std::vector<double>> identity_matrix(const int n) {
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      I[i][i] = 1.0;
    }
    return I;
  }

  static std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& A) {
    if (A.empty()) return {};
    std::vector<std::vector<double>> At(A[0].size(), std::vector<double>(A.size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i) {
      for (size_t j = 0; j < A[i].size(); ++j) {
        At[j][i] = A[i][j];
      }
    }
    return At;
  }

  static std::vector<std::vector<double>> matmul(
      const std::vector<std::vector<double>>& A,
      const std::vector<std::vector<double>>& B) {
    if (A.empty() || B.empty()) return {};
    const size_t n = A.size();
    const size_t m = B[0].size();
    const size_t inner = A[0].size();
    std::vector<std::vector<double>> C(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < n; ++i) {
      for (size_t k = 0; k < inner; ++k) {
        const double aik = A[i][k];
        if (aik == 0.0) continue;
        for (size_t j = 0; j < m; ++j) {
          C[i][j] += aik * B[k][j];
        }
      }
    }
    return C;
  }

  static std::vector<double> matvec(
      const std::vector<std::vector<double>>& A,
      const std::vector<double>& x) {
    if (A.empty()) return {};
    std::vector<double> y(A.size(), 0.0);
    for (size_t i = 0; i < A.size(); ++i) {
      for (size_t j = 0; j < A[i].size(); ++j) {
        y[i] += A[i][j] * x[j];
      }
    }
    return y;
  }

  static void jacobi_eigendecomposition(
      const std::vector<std::vector<double>>& input,
      std::vector<std::vector<double>>& eigenvectors,
      std::vector<double>& eigenvalues,
      const int max_iter = 200,
      const double tol = 1e-12) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
      eigenvectors.clear();
      eigenvalues.clear();
      return;
    }
    std::vector<std::vector<double>> A = input;
    eigenvectors = identity_matrix(n);

    for (int iter = 0; iter < max_iter * n * n; ++iter) {
      int p = 0;
      int q = 1;
      double max_offdiag = 0.0;

      for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
          const double val = std::fabs(A[i][j]);
          if (val > max_offdiag) {
            max_offdiag = val;
            p = i;
            q = j;
          }
        }
      }

      if (max_offdiag < tol) break;

      const double app = A[p][p];
      const double aqq = A[q][q];
      const double apq = A[p][q];

      if (std::fabs(apq) < tol) continue;

      const double tau = (aqq - app) / (2.0 * apq);
      const double t = (tau >= 0.0)
                           ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                           : 1.0 / (tau - std::sqrt(1.0 + tau * tau));
      const double c = 1.0 / std::sqrt(1.0 + t * t);
      const double s = t * c;

      for (int i = 0; i < n; ++i) {
        if (i == p || i == q) continue;
        const double aip = A[i][p];
        const double aiq = A[i][q];
        A[i][p] = c * aip - s * aiq;
        A[p][i] = A[i][p];
        A[i][q] = c * aiq + s * aip;
        A[q][i] = A[i][q];
      }

      A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
      A[p][q] = 0.0;
      A[q][p] = 0.0;

      for (int i = 0; i < n; ++i) {
        const double vip = eigenvectors[i][p];
        const double viq = eigenvectors[i][q];
        eigenvectors[i][p] = c * vip - s * viq;
        eigenvectors[i][q] = s * vip + c * viq;
      }
    }

    eigenvalues.resize(n);
    for (int i = 0; i < n; ++i) {
      eigenvalues[i] = A[i][i];
    }
  }

  int decode_mode_index(const int mode_id) const {
    if (mode_id < 0) return mode_id;
    if (mode_id == 0) return 0;
    if (mode_id % 2 == 1) return (mode_id + 1) / 2;
    return -(mode_id / 2);
  }

  double mode_center(const int mode_id) const {
    return static_cast<double>(decode_mode_index(mode_id)) * center_spacing;
  }

  double minimal_image(const double delta) const {
    if (!pbc) return delta;
    return delta - period * std::round(delta / period);
  }

  std::vector<std::vector<double>> compute_gram_matrix() const {
    std::vector<std::vector<double>> G(n_modes, std::vector<double>(n_modes, 0.0));
    const double a = domain_min;
    const double b = domain_max;
    const double sqrt2 = std::sqrt(2.0);

    for (int i = 0; i < n_modes; ++i) {
      const bool i_const = (i == 0);
      const double ci = mode_center(i);
      for (int j = 0; j <= i; ++j) {
        const bool j_const = (j == 0);
        const double cj = mode_center(j);
        double gij = 0.0;

        if (i_const && j_const) {
          gij = (b - a) / 2.0;
        } else if (i_const || j_const) {
          if (pbc) {
            const double g1 = sigma * std::sqrt(2.0 * M_PI) *
                              std::erf(period / (2.0 * std::sqrt(2.0) * sigma));
            gij = g1 / sqrt2;
          } else {
            const double c = i_const ? cj : ci;
            const double inv = 1.0 / (std::sqrt(2.0) * sigma);
            const double z_hi = (b - c) * inv;
            const double z_lo = (a - c) * inv;
            const double g1 = sigma * std::sqrt(M_PI / 2.0) * (std::erf(z_hi) - std::erf(z_lo));
            gij = g1 / sqrt2;
          }
        } else {
          if (pbc) {
            const double delta = minimal_image(ci - cj);
            const double pref = std::exp(-0.25 * (delta / sigma) * (delta / sigma));
            const double window = sigma * std::sqrt(M_PI) * std::erf(period / (2.0 * sigma));
            gij = pref * window;
          } else {
            const double delta = ci - cj;
            const double mid = 0.5 * (ci + cj);
            const double pref = std::exp(-0.25 * (delta / sigma) * (delta / sigma));
            const double erf_term = std::erf((b - mid) / sigma) - std::erf((a - mid) / sigma);
            gij = pref * (sigma * std::sqrt(M_PI) * 0.5) * erf_term;
          }
        }
        G[i][j] = gij;
        G[j][i] = gij;
      }
    }

    return G;
  }

  std::vector<std::vector<double>> compute_whitening_matrix() const {
    std::vector<std::vector<double>> G = compute_gram_matrix();
    std::vector<std::vector<double>> Q;
    std::vector<double> evals;
    jacobi_eigendecomposition(G, Q, evals);
    const int n = static_cast<int>(evals.size());
    std::vector<std::vector<double>> Dinv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
      const double lam = std::max(evals[i], whitening_regularization);
      Dinv[i][i] = 1.0 / std::sqrt(lam);
    }
    return matmul(matmul(Q, Dinv), transpose(Q));
  }

  std::vector<double> apply_whitening(const std::vector<double>& feat) const {
    if (!whiten) return feat;
    std::vector<double> out(n_modes, 0.0);
    for (int j = 0; j < n_modes; ++j) {
      for (int i = 0; i < n_modes; ++i) {
        out[j] += feat[i] * whitening_matrix[i][j];
      }
    }
    return out;
  }

  std::vector<double> gaussian_feature(const double x) const {
    std::vector<double> feat(n_modes, 0.0);
    feat[0] = 1.0 / std::sqrt(2.0);
    for (int m = 1; m < n_modes; ++m) {
      const double c = mode_center(m);
      const double delta = minimal_image(x - c);
      feat[m] = std::exp(-0.5 * (delta / sigma) * (delta / sigma));
    }
    return apply_whitening(feat);
  }

  std::vector<double> grad_gaussian_feature(const double x) const {
    std::vector<double> feat(n_modes, 0.0);
    feat[0] = 0.0;
    for (int m = 1; m < n_modes; ++m) {
      const double c = mode_center(m);
      const double delta = minimal_image(x - c);
      const double val = std::exp(-0.5 * (delta / sigma) * (delta / sigma));
      feat[m] = -(delta / (sigma * sigma)) * val;
    }
    return apply_whitening(feat);
  }

  std::vector<double> compute_basis_integrals() const {
    std::vector<double> raw(n_modes, 0.0);
    raw[0] = (domain_max - domain_min) / std::sqrt(2.0);
    if (n_modes > 1) {
      if (pbc) {
        const double integ = sigma * std::sqrt(2.0 * M_PI) *
                             std::erf(period / (2.0 * std::sqrt(2.0) * sigma));
        for (int m = 1; m < n_modes; ++m) raw[m] = integ;
      } else {
        const double inv = 1.0 / (std::sqrt(2.0) * sigma);
        for (int m = 1; m < n_modes; ++m) {
          const double c = mode_center(m);
          const double z_hi = (domain_max - c) * inv;
          const double z_lo = (domain_min - c) * inv;
          raw[m] = sigma * std::sqrt(M_PI / 2.0) * (std::erf(z_hi) - std::erf(z_lo));
        }
      }
    }
    if (!whiten) return raw;

    std::vector<double> projected(n_modes, 0.0);
    for (int j = 0; j < n_modes; ++j) {
      for (int i = 0; i < n_modes; ++i) {
        projected[j] += whitening_matrix[i][j] * raw[i];
      }
    }
    return projected;
  }

  std::vector<double> contract_leaf_message(
      const std::vector<double>& core_eval,
      const std::vector<std::vector<double>>& c_matrix) const {
    std::vector<double> P(c_matrix[0].size(), 0.0);
    const size_t n_row = std::min(core_eval.size(), c_matrix.size());
    for (size_t i = 0; i < n_row; ++i) {
      const auto& row = c_matrix[i];
      for (size_t j = 0; j < row.size(); ++j) {
        P[j] += core_eval[i] * row[j];
      }
    }
    return P;
  }

  double contract_root(const TensorCollection& eval_msg) const {
    const int l = 0;
    const int k = 1;
    const auto& c_matrix = collection.get2D(std::to_string(k) + "_" + std::to_string(l));
    const auto& P1 = eval_msg.get2D(std::to_string(2 * k - 1) + "_" + std::to_string(l + 1));
    const auto& P2 = eval_msg.get2D(std::to_string(2 * k) + "_" + std::to_string(l + 1));
    double rho = 0.0;
    for (size_t i = 0; i < P1[0].size(); ++i) {
      for (size_t j = 0; j < P2[0].size(); ++j) {
        rho += P1[0][i] * P2[0][j] * c_matrix[i][j];
      }
    }
    return rho;
  }

  double compute_normalization_constant_raw() {
    TensorCollection eval_msg;
    int l = depth;
    for (int k = 1; k <= (1 << l); ++k) {
      std::vector<double> P = masked_leaf_node(k, l);
      TensorData tensor_tmp;
      tensor_tmp.shape = {1, static_cast<uint32_t>(P.size())};
      tensor_tmp.data2D.clear();
      tensor_tmp.data2D.push_back(P);
      eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
    }

    for (l = depth - 1; l > 0; --l) {
      for (int k = 1; k <= (1 << l); ++k) {
        std::vector<double> P = middle_node(k, l, eval_msg);
        TensorData tensor_tmp;
        tensor_tmp.shape = {1, static_cast<uint32_t>(P.size())};
        tensor_tmp.data2D.push_back(P);
        eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
      }
    }
    return contract_root(eval_msg);
  }

  public:
  void load(
      TensorCollection& collection,
      int depth,
      int deg,
      int d_true,
      double sigma,
      double center_spacing,
      bool pbc,
      double period,
      double domain_min,
      double domain_max,
      bool whiten,
      double whitening_regularization,
      bool normalize) {
    this->collection = collection;
    this->depth = depth;
    this->deg = deg;
    this->d_true = d_true;
    this->d_pad = 1 << depth;
    this->n_modes = 2 * deg + 1;
    this->sigma = sigma;
    this->center_spacing = center_spacing;
    this->pbc = pbc;
    this->period = period;
    this->domain_min = domain_min;
    this->domain_max = domain_max;
    this->whiten = whiten;
    this->whitening_regularization = whitening_regularization;
    this->normalize = normalize;
    this->normalization_constant = 1.0;

    if (this->sigma <= 0.0) {
      throw std::runtime_error("sigma must be positive");
    }
    if (this->domain_max <= this->domain_min) {
      throw std::runtime_error("domain must satisfy DOMAIN_MAX > DOMAIN_MIN");
    }
    if (this->center_spacing <= 0.0) {
      this->center_spacing = (this->domain_max - this->domain_min) / static_cast<double>(this->n_modes);
    }
    if (this->period <= 0.0) {
      this->period = this->domain_max - this->domain_min;
    }
    if (this->pbc && this->period <= 0.0) {
      throw std::runtime_error("period must be positive when pbc is enabled");
    }
    if (this->whitening_regularization <= 0.0) {
      this->whitening_regularization = 1e-10;
    }

    if (this->whiten) {
      this->whitening_matrix = compute_whitening_matrix();
    } else {
      this->whitening_matrix = identity_matrix(this->n_modes);
    }
    this->basis_integrals = compute_basis_integrals();

    if (this->normalize) {
      this->normalization_constant = compute_normalization_constant_raw();
      if ((!std::isfinite(this->normalization_constant)) || this->normalization_constant <= 0.0) {
        throw std::runtime_error("Invalid normalization constant in Rho::load");
      }
    }
  }

  std::vector<double> leaf_node(const int k, const int l, const double x) {
    std::vector<double> core_eval = gaussian_feature(x);
    const auto& c_matrix = collection.get2D(std::to_string(k) + "_" + std::to_string(l));
    return contract_leaf_message(core_eval, c_matrix);
  }
  std::vector<double> grad_leaf_node(const int k, const int l, const double x) {
    std::vector<double> core_eval = grad_gaussian_feature(x);
    const auto& c_matrix = collection.get2D(std::to_string(k) + "_" + std::to_string(l));
    return contract_leaf_message(core_eval, c_matrix);
  }
  std::vector<double> middle_node(const int k, const int l, TensorCollection& eval_msg) {
  const auto& c_matrix = collection.get3D(std::to_string(k) + "_" + std::to_string(l));
  const auto& P1 = eval_msg.get2D(std::to_string(2*k-1) + "_" + std::to_string(l+1));
  const auto& P2 = eval_msg.get2D(std::to_string(2*k) + "_" + std::to_string(l+1));
  std::vector<double> P(c_matrix.size(), 0.0);
  for (size_t i = 0; i < P.size(); ++i) {
    for (size_t j = 0; j < P1[0].size(); ++j) {
      for (size_t kk = 0; kk < P2[0].size(); ++kk) {
        P[i] += P1[0][j] * P2[0][kk] * c_matrix[i][j][kk];
      }
    }
  }
  return P;
  }

  double compute_rho(const std::vector<double>& variable){
    int l,k;
    TensorCollection eval_msg;
    l = depth;
    for (k = 1; k <= (1 << l); ++k) {
      std::vector<double> P;
      if (k <= d_true) {
        P = leaf_node(k, l, variable[k-1]);
      } else {
        // padding 的维度：做 marginal（积分掉）
        P = masked_leaf_node(k, l);
      }

      TensorData tensor_tmp;
      tensor_tmp.shape = {1, static_cast<uint32_t>(P.size())};
      tensor_tmp.data2D.clear();
      tensor_tmp.data2D.push_back(P);
      eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
    }


    for (l = depth-1; l > 0; --l) {
      for (int k = 1; k <= (1 << l); ++k) {
        std::vector<double> P = middle_node(k, l, eval_msg);
        TensorData tensor_tmp;
        std::vector<uint32_t> shape = {1, static_cast<unsigned int>(P.size())};
        tensor_tmp.shape = shape;
        tensor_tmp.data2D.push_back(P);
        eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
      }
    }

    double rho = contract_root(eval_msg);
    if (normalize) {
      rho /= normalization_constant;
    }
    return rho;
  }

  std::vector<double> masked_leaf_node(const int k, const int l) {
    const auto& c_matrix = collection.get2D(std::to_string(k) + "_" + std::to_string(l));
    std::vector<double> P(c_matrix[0].size(), 0.0);
    const size_t n_row = std::min(c_matrix.size(), basis_integrals.size());
    for (size_t j = 0; j < P.size(); ++j) {
      for (size_t i = 0; i < n_row; ++i) {
        P[j] += basis_integrals[i] * c_matrix[i][j];
      }
    }
    return P;
  }

  double compute_grad_rho(const std::vector<double>& variable, int k_dir){
    int l,k;
    TensorCollection eval_msg;
    l = depth;
    for (k = 1; k <= (1 << l); ++k) {
      std::vector<double> P;

      if (k <= d_true) {
        P = leaf_node(k, l, variable[k-1]);
        if (k == k_dir) {
          P = grad_leaf_node(k, l, variable[k-1]);
        }
      } else {
        // padding 维度仍然 marginal 掉：常数 message
        P = masked_leaf_node(k, l);
      }

      TensorData tensor_tmp;
      tensor_tmp.shape = {1, static_cast<uint32_t>(P.size())};
      tensor_tmp.data2D.clear();
      tensor_tmp.data2D.push_back(P);
      eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
    }


    for (l = depth-1; l > 0; --l) {
      for (int k = 1; k <= (1 << l); ++k) {
        std::vector<double> P = middle_node(k, l, eval_msg);
        TensorData tensor_tmp;
        std::vector<uint32_t> shape = {1, static_cast<unsigned int>(P.size())};
        tensor_tmp.shape = shape;
        tensor_tmp.data2D.push_back(P);
        eval_msg.update(std::to_string(k) + "_" + std::to_string(l), tensor_tmp);
      }
    }

    double rho = contract_root(eval_msg);
    if (normalize) {
      rho /= normalization_constant;
    }
    return rho;
  }
  
};

class MetaTensor : public Bias {
  int num_collections;
  int depth;
  int deg;
  double kbt_;
  double eps;
  double temp;
  int d_true_;
  int d_pad_;
  std::vector<std::string> collection_dirs;
  std::vector<TensorCollection> collections_;
  std::vector<Rho> rhos_;
  double tau_factor;
  double alpha;
  double sigma_;
  double center_spacing_;
  bool pbc_;
  double period_;
  double domain_min_;
  double domain_max_;
  bool whiten_;
  double whitening_regularization_;
  bool normalize_;
  int use_bin_params_;
  Value* valueForce2;
public:
  explicit MetaTensor(const ActionOptions&);
  void calculate() override;
  // std::vector<double> fourier_function(const double x, const int deg);
  // std::vector<double> middle_node(const int k, const int l, TensorCollection& eval_msg);
  // std::vector<double> leaf_node(const int k, const int l, const double x);
  // double compute_rho(const std::vector<double>& variable);
  // double compute_grad_rho(const std::vector<double>& variable, int k_dir);
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(MetaTensor,"METATENSOR")

void MetaTensor::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory", "NUMCOLLECTIONS", "1", "Number of tensor collections");
  keys.add("compulsory", "EPS", "1", "Number of tensor collections");
  keys.add("compulsory","TEMP","300","the system temperature - this is only needed if you are doing well-tempered metadynamics");
  keys.add("compulsory","TAU_FACTOR","0.5","tau factor of eps"); 
  keys.add("compulsory","ALPHA","1.0","exponent factor in log(rho_eff^alpha)");
  keys.add("compulsory","SIGMA","0.2","Gaussian kernel width");
  keys.add("compulsory","CENTER_SPACING","-1.0","Gaussian center spacing; <=0 uses (DOMAIN_MAX-DOMAIN_MIN)/(2*deg+1)");
  keys.add("compulsory","PBC","1","Use periodic minimal-image distance in Gaussian basis (1/0)");
  keys.add("compulsory","PERIOD","2.0","Period used when PBC=1");
  keys.add("compulsory","DOMAIN_MIN","-1.0","Lower bound of basis domain");
  keys.add("compulsory","DOMAIN_MAX","1.0","Upper bound of basis domain");
  keys.add("compulsory","WHITEN","1","Apply Gram whitening to Gaussian basis (1/0)");
  keys.add("compulsory","WHITENING_REGULARIZATION","1e-10","Eigenvalue floor for whitening matrix");
  keys.add("compulsory","NORMALIZE","1","Normalize density by full-domain integral (1/0)");
  // PLUMED 2.9 Keywords::add(type,key,default,desc) only allows type=compulsory/hidden.
  // Keep default behavior via compulsory keyword with default value "1".
  keys.add("compulsory","USE_BIN_PARAMS","1","If 1, read sigma/center_spacing from COLLECTIONDIRS/basis_params.dat when present");
  keys.add("optional", "COLLECTIONDIRS", "Directories for tensor collections");
  keys.addOutputComponent("force2","default","the instantaneous value of the squared force due to this bias potential");
}

MetaTensor::MetaTensor(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao)
{
  parse("NUMCOLLECTIONS", num_collections);
  parse("EPS",eps);
  collection_dirs.resize(num_collections);
  parseVector("COLLECTIONDIRS", collection_dirs);
  parse("TAU_FACTOR",tau_factor);
  parse("ALPHA",alpha);
  sigma_ = 0.2;
  center_spacing_ = -1.0;
  period_ = 2.0;
  domain_min_ = -1.0;
  domain_max_ = 1.0;
  whitening_regularization_ = 1e-10;
  int pbc_int = 1;
  int whiten_int = 1;
  int normalize_int = 1;
  use_bin_params_ = 1;
  parse("SIGMA", sigma_);
  parse("CENTER_SPACING", center_spacing_);
  parse("PBC", pbc_int);
  parse("PERIOD", period_);
  parse("DOMAIN_MIN", domain_min_);
  parse("DOMAIN_MAX", domain_max_);
  parse("WHITEN", whiten_int);
  parse("WHITENING_REGULARIZATION", whitening_regularization_);
  parse("NORMALIZE", normalize_int);
  parse("USE_BIN_PARAMS", use_bin_params_);
  pbc_ = (pbc_int != 0);
  whiten_ = (whiten_int != 0);
  normalize_ = (normalize_int != 0);
  checkRead();
  temp=0.0;
  parse("TEMP",temp);
  if(temp>0.0) kbt_=plumed.getAtoms().getKBoltzmann()*temp;
  else kbt_=plumed.getAtoms().getKbT();
  collections_.resize(num_collections);
  int nv = getNumberOfArguments();
  d_true_ = nv;

  // 方案A：pad到最近的2^L
  depth = static_cast<int>(std::ceil(std::log2(static_cast<double>(nv))));
  d_pad_ = 1 << depth;

  std::cout << "True dim nv = " << d_true_ 
            << ", depth = " << depth 
            << ", padded dim = " << d_pad_ << std::endl;

  if (d_true_ > d_pad_) {
    plumed_merror("d_true > d_pad : impossible");
  }


  std::cout << "Depth: " << depth << std::endl;
  std::cout << "Number of collections: " << num_collections << std::endl;
  // if (collection_dirs.size() != num_collections) {
  //   plumed_merror("Mismatch between NUM_COLLECTIONS and COLLECTION_DIRS size");
  // }
  // 建议保留一次 resize 就够了
  collections_.resize(num_collections);

  // （可选但推荐）检查 collection_dirs 是否给够
  if (static_cast<int>(collection_dirs.size()) != num_collections) {
    plumed_merror("Mismatch between NUMCOLLECTIONS and COLLECTIONDIRS size");
  }

  if (use_bin_params_ != 0) {
    std::string source_file;
    double loaded_sigma = sigma_;
    double loaded_center_spacing = center_spacing_;
    if (load_basis_params_from_dirs(collection_dirs, loaded_sigma, loaded_center_spacing, source_file)) {
      sigma_ = loaded_sigma;
      center_spacing_ = loaded_center_spacing;
      std::cout << "Loaded Gaussian basis params from " << source_file
                << ": sigma=" << sigma_
                << ", center_spacing=" << center_spacing_ << std::endl;
    } else {
      std::cout << "No basis_params.dat found in COLLECTIONDIRS; keep input/default SIGMA/CENTER_SPACING."
                << std::endl;
    }
  }

  for (int i = 0; i < num_collections; ++i) {
    const std::string& dir = collection_dirs[i];
    if (dir.empty()) {
      plumed_merror("COLLECTIONDIRS contains an empty path. Please set COLLECTIONDIRS=... correctly.");
    }

    for (int l = 0; l <= depth; ++l) {
      int nnode = 1 << l;
      for (int k = 1; k <= nnode; ++k) {
        const std::string key = std::to_string(k) + "_" + std::to_string(l);
        const std::string fn  = dir + "/" + key + ".bin";
        collections_[i].load(key, fn);
      }
    }
  }



  TensorCollection collection = collections_[0];
  // Access 2D tensor
  const auto& matrix = collection.get2D("1_0");
  std::cout << "2D Tensor (1,0):\n";
  std::cout << "Size: " << matrix.size() << "x" << matrix[0].size() << std::endl;

  // for (const auto& row : matrix) {
  //     for (double val : row) std::cout << val << " ";
  //     std::cout << "\n";
  // }

  // Access 3D tensor
  if (depth >= 2) {
    const auto& tensor3D = collection.get3D("1_1");
    std::cout << "\n3D Tensor (1,1):\n";
    for (size_t d = 0; d < tensor3D.size(); ++d) {
        // std::cout << "Depth " << d << ":\n";
        // for (const auto& row : tensor3D[d]) {
        //     for (double val : row) std::cout << val << " ";
        //     std::cout << "\n";
        // }
    }
  }

  const auto& matrix2 = collection.get2D("1_" +std::to_string(depth) );
  std::cout << "2D Tensor (1,depth):\n";
  std::cout << "Size: " << matrix2.size() << "x" << matrix2[0].size() << std::endl;
  // for (const auto& row : matrix2) {
  //     for (double val : row) std::cout << val << " ";
  //     std::cout << "\n";
  // }
  deg = (matrix2.size() - 1)/2;
  std::cout << "Degree: " << deg << std::endl;
  if (domain_max_ <= domain_min_) {
    plumed_merror("DOMAIN_MAX must be larger than DOMAIN_MIN");
  }
  if (sigma_ <= 0.0) {
    plumed_merror("SIGMA must be positive");
  }
  if (center_spacing_ <= 0.0) {
    center_spacing_ = (domain_max_ - domain_min_) / static_cast<double>(2 * deg + 1);
  }
  if (period_ <= 0.0) {
    period_ = domain_max_ - domain_min_;
  }
  if (pbc_ && period_ <= 0.0) {
    plumed_merror("PERIOD must be positive when PBC=1");
  }
  std::cout << "Gaussian parameters: sigma=" << sigma_
            << ", center_spacing=" << center_spacing_
            << ", pbc=" << (pbc_ ? 1 : 0)
            << ", period=" << period_
            << ", domain=[" << domain_min_ << "," << domain_max_ << "]"
            << ", whiten=" << (whiten_ ? 1 : 0)
            << ", normalize=" << (normalize_ ? 1 : 0)
            << std::endl;

  for (int i = 0; i < num_collections; ++i) {
    Rho rho;
    rho.load(
        collections_[i],
        depth,
        deg,
        d_true_,
        sigma_,
        center_spacing_,
        pbc_,
        period_,
        domain_min_,
        domain_max_,
        whiten_,
        whitening_regularization_,
        normalize_);
    rhos_.push_back(rho);
  }

  addComponent("force2");
  componentIsNotPeriodic("force2");
  valueForce2=getPntrToComponent("force2");
}
static inline double softplus(double x) {
  // stable softplus
  if (x > 50.0) return x;
  if (x < -50.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

static inline double sigmoid(double x) {
  if (x >= 0.0) {
    double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  } else {
    double z = std::exp(x);
    return z / (1.0 + z);
  }
}

void MetaTensor::calculate() {
  // std::vector<double> variable = {0.4, 0.6, 0.8, 1.0};
  // double rho = rhos_[0].compute_rho(variable);
  // std::cout << "Rho: " << rho << std::endl;
  // double grad_rho = rhos_[0].compute_grad_rho(variable, 1);
  // std::cout << "Grad Rho: " << grad_rho << std::endl;
  // variable = {0.1, 0.5, 0.5, 0.5};
  // rho = rhos_[0].compute_rho(variable);
  // grad_rho = rhos_[0].compute_grad_rho(variable, 1);
  // std::cout << "Rho: " << rho << std::endl;
  // std::cout << "Grad Rho: " << grad_rho << std::endl;
  // variable = {0.2, 0.5, 0.5, 0.5};
  // rho = rhos_[0].compute_rho(variable);
  // grad_rho = rhos_[0].compute_grad_rho(variable, 1);
  // std::cout << "Rho: " << rho << std::endl;
  // std::cout << "Grad Rho: " << grad_rho << std::endl;
  // variable = {0.3, 0.5, 0.5, 0.5};
  // rho = rhos_[0].compute_rho(variable);
  // grad_rho = rhos_[0].compute_grad_rho(variable, 1);
  // std::cout << "Rho: " << rho << std::endl;
  // std::cout << "Grad Rho: " << grad_rho << std::endl;
  double ene=0.0;
  double totf2=0.0;

  int nv = getNumberOfArguments();
  std::vector<double> variable(d_pad_, 0.0);

  // 前116维来自 ARG
  for (int i = 0; i < nv; ++i) {
    variable[i] = getArgument(i) / M_PI;
  }

  // 后12维随便填0就行（因为在Rho里会被masked_leaf_node积分掉）
  for (int i = nv; i < d_pad_; ++i) {
    variable[i] = 0.0;
  }

  std::vector<double> rho(num_collections, 0.0);
  double tau = tau_factor * eps;  // 平滑宽度，可调：0.1~0.5 * eps
  for (int j = 0; j < num_collections; ++j) {
    double rho_raw = rhos_[j].compute_rho(variable);
    if (!std::isfinite(rho_raw)) rho_raw = 0.0;

    rho[j] = rho_raw;  // 保存清洗后的 rho，后面算力也用它

    double t = rho_raw / tau;
    double rho_eff = eps + tau * softplus(t);

    ene += kbt_ * alpha * std::log(rho_eff);
  }

  std::vector<double> grad_rho_test(getNumberOfArguments(),0.0);
  for(unsigned i=0; i<getNumberOfArguments(); ++i) {
    std::vector<double> grad_rho(num_collections, 0.0);
    for (int j = 0; j < num_collections; ++j) {
      grad_rho[j] = rhos_[j].compute_grad_rho(variable, i+1);
    }
    grad_rho_test[i] = grad_rho[0];
    double f = 0.0;
    for (int j = 0; j < num_collections; ++j) {
      double t = rho[j] / tau;
      double sigma = sigmoid(t);                 // drho_eff/drho
      double rho_eff = eps + tau * softplus(t);  // rho_eff
      // f = - d/dx [ kBT alpha log(rho_eff) ]
      f -= kbt_ * alpha * (sigma / rho_eff) * grad_rho[j];
    }
    
    f = f / M_PI;        // 你原来就有的链式 (arg/pi)
    setOutputForce(i, f);
    totf2 += f*f;
  }
  // std::cout << "At " << std::endl;
  // for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  // std::cout << variable[i] << " " ;
  // }
  // std::cout<< std::endl;
  // std::cout << "Rho " << rho[0] << std::endl;
  // std::cout <<  "Gradient Rho" << std::endl;
  // for(unsigned i=0; i<getNumberOfArguments(); ++i) {
  // std::cout << grad_rho_test[i] << " " ;
  // }
  // std::cout<< std::endl;
  // rho.resize(10);
  // for (int j = 0; j < rho.size(); ++j) {
  //   rho[j] = j-5;
  //   ene = 0.0;
  //   if (rho[j]> eps) {
  //     ene += kbt_/rho[j];
  //   } else if (rho[j] > 0.0) {
  //     ene += kbt_*(2*b2*rho[j] + b1);
  //   } else {
  //     ene += (-2.0*c2*rho[j] - c1)/(c2*std::pow(rho[j], 2) + c1*rho[j] + c0);
  //   } 
  //   std::cout << "Rho: " << rho[j] << std::endl;
  //   std::cout << "Energy': " << ene << std::endl;
  // }

  setBias(ene);
  valueForce2->set(totf2);
}

}

}
