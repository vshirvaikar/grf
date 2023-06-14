/*-------------------------------------------------------------------------------
  This file is part of generalized random forest (grf).

  grf is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  grf is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with grf. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/

#include <algorithm>
#include <cmath>
#include <armadillo>

#include "InstrumentalGLM.h"

namespace grf {

    InstrumentalGLM::InstrumentalGLM(size_t dummy){ // constructor
        this->counter = new size_t[dummy];
    }

    InstrumentalGLM::~InstrumentalGLM(){ // destructor
        if (counter != nullptr) {
            delete[] counter;
        }
    }

    double InstrumentalGLM::dummy(){
        double output = (double) rand() / RAND_MAX;
        return output;
    }

    arma::colvec InstrumentalGLM::variance(std::string family, arma::colvec mu){
        if(family == "logistic"){
            return mu % (1.0 - mu);
        } else if(family == "poisson"){
            return mu;
        }
        return arma::ones<arma::colvec>(mu.n_elem);
    }

    arma::colvec InstrumentalGLM::invlink(std::string family, arma::colvec eta){
        if(family == "logistic"){
            return (arma::exp(eta) / (1.0 + arma::exp(eta)));
        } else if(family == "poisson"){
            return arma::exp(eta);
        }
        return eta;
    }

    arma::colvec InstrumentalGLM::invlink_prime(std::string family, arma::colvec eta){
        if(family == "logistic"){
            return arma::exp(eta) / arma::square(arma::exp(eta) + 1.0);
        } else if(family == "poisson"){
            return arma::exp(eta);
        }
        return arma::ones<arma::colvec>(eta.n_elem);
    }

    double InstrumentalGLM::glm_fit(arma::mat X, arma::colvec y, std::string family, int maxit, double tol) {

        const int n_cols = X.n_cols;
        const int n_rows = X.n_rows;
        arma::mat Q, R;
        arma::colvec s = arma::zeros<arma::colvec>(n_cols);
        arma::colvec s_old;
        arma::colvec eta = arma::ones<arma::colvec>(n_rows);
        arma::qr_econ(Q, R, X);
        arma::mat V;

        for (int i = 0; i < maxit; i++) {
            s_old = s;
            const arma::colvec mu = invlink(family, eta);
            const arma::colvec mu_p = invlink_prime(family, eta);
            const arma::colvec z = eta + (y - mu) / mu_p;
            const arma::colvec W = arma::square(mu_p) / variance(family, mu);
            const arma::mat C = arma::chol(Q.t() * (Q.each_col() % W));
            const arma::colvec s1 = arma::solve(arma::trimatl(C.t()), Q.t() * (W % z));
            s = arma::solve(arma::trimatu(C), s1);
            eta = Q * s;

            const bool is_converged = std::sqrt(arma::accu(arma::square(s - s_old))) < tol;
            if (is_converged) {
                // std::cout << "iteration " << i << std::endl;
                arma::mat Wd = arma::diagmat(W);
                arma::mat combo = X.t() * Wd * X;
                if(arma::det(combo) < 0.0001){
                    return 0;
                }
                V = arma::inv(combo);
                break;
            }
        }

        arma::colvec coeffs = arma::solve(arma::trimatu(R), Q.t() * eta);
        arma::colvec stderrs = sqrt(V.diag());
        arma::colvec tstats = coeffs / stderrs;
        return abs(tstats(tstats.n_rows - 1));
    }

} // namespace grf