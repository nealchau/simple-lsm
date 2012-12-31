/*! @file */

/** \mainpage This code implements the Longstaff-Schwartz LSM.
 * This file computes the Longstaff-Schwartz Least-Squares Monte
 * Carlo method, along with classes to implement Asian and Bermudan
 * puts and calls.  We generate stock paths using geometric Brownian
 * motion, and we include Poisson jumps.  We calculate the hedges:
 * delta, gamma, vega, vega-of-vega, tau, tau-of-tau (bernanke).
 * \section LSM
 * The Monte Carlo method is a form of simulation and dynamic
 * programming.  We start at the final timestep and then go
 * backwards in time through the timesteps.  At each timestep we
 * regress our current stock price against the expected value of
 * holding the option and compare the regressed value to the
 * immediate exercise value and choose the larger value.
 * \section Stockpaths
 * Stock paths are classically modelled as geometric Brownian
 * motion.  We also generate Poisson jumped stock paths, and we
 * "bump" stock paths to form Greeks using Monte Carlo "finite
 * difference"
 * \section Greeks
 * We compute Greeks by bumping paths up and down in initial stock
 * price, in timestep size (tau), and in volatility (vega).  We form
 * first- and second-order Greeks by using centered finite
 * difference formulas.
 */

#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "bs.h"

using namespace std;
using namespace boost::numeric::ublas;
namespace ublas = boost::numeric::ublas;

/*! 
 * \brief Boost UBLAS vector of doubles
 *
 * We use the Boost C++ UBLAS library to create vectors of doubles.
 */
typedef boost::numeric::ublas::vector<double> vec;  
/*! 
 * \brief Boost UBLAS matrix of doubles
 *
 * We use the Boost C++ UBLAS library to create matrix of doubles.
 */
typedef boost::numeric::ublas::matrix<double> mat;

/*! 
 * \brief This is a pure virtual base class for all option payoffs
 * 
 * As a pure virtual base, you do not actually create instances of this class.
 * Instead you derive classes (specifically, particular option payoffs) from it.  It
 * creates a "guaranteed" set of member functions that any code can call on any
 * derived option payoff object.  This means any class you derive from payofffunc
 * must specify what the immediate exercise value is, and for any object of class
 * payofffunc, you can find the immediate exercise value.
 */

class payofffunc
{
public:
    /*! 
     * \brief This function is pure virtual and gives the option payoff on a
     * vector_expression.
     *
     * Vector_expressions can be, for example, slices or rows of matrices.  The matrix of
     * stock payoffs contains rows which represent individual stock paths.  This
     * function will take any vector_expression such as a row of a matrix and compute
     * the immediate exercise option value.
     *
     * @param stockpath This is a vector_expression such as a row, representing a
     * stock path, on which we want to compute the immediate option value
     * @param tindex This is the time at which we are computing the option's
     * immediate exercise value
     * @return Returns the immediate exercise value of the option on the path at the
     * time
     */
    virtual double immediate(const vector_expression<vec> &stockpath, int tindex) = 0;
    /*! 
     * \brief This function is pure virtual and gives the option payoff on a vec
     * 
     * A vec is an actual UBLAS vector, and not a vector_expression such as a 
     * row of a matrix.  This function is
     * typically called by the vector_expression function to evaluate the immediate
     * option value.  
     * 
     * @param stockpath This vec contains the stock path on which we are evaluating
     * the immediate option value
     * @param tindex This is the time at which we are finding the option's immediate
     * exercise value
     * @return Returns the immediate exercise value on the path at the time
     */
    virtual double immediate(const vec &stockpath, int tindex) = 0;
    
};


/*! 
 * \brief Computes the payoff of a put, used in European and Bermudan puts
 *
 * This class implements the payofffunc base class interface as a put.  
 */
class putpayoff : public payofffunc 
{
public:
    /*! 
     * \brief Constructor, stores the put strike in a member
     * 
     * A constructor is a member function that has the same name as the class its in,
     * and it prepares the member data.  In this case, the constructor stores the
     * strike price in a data member.
     */

    putpayoff(double K_) : K(K_) { } 
    
    /*! 
     * \brief computes the put price on the stock path vector
     *
     * In this function, we compute the put price on the
     * vector of stock prices.  this is the function which
     * computes the immediate exercise value of a put as
     * "promised" by our pure virtual base class
     * payofffunc
     * 
     * @param stockpath is a vector of stock prices 
     * @param tindex is the time at which we find the immediate
     * value 
     * @return the put price
     */

    double immediate(const vec &stockpath, int tindex)
    {
        return max(0., K-stockpath(tindex));
    }

    /*! 
     * \brief computes the put price on any stock path
     * vector_expression
     *
     * In this function, we compute the put price on a
     * vector_expression of stock prices.  
     * this function prepares an actual vec and calls the
     * overloaded (other) version which operates on vec.
     * 
     * @param stockpath is a vector_expression of stock prices 
     * @param tindex is the time at which we find the immediate
     * value 
     * @return the put price
     */

    double immediate(const vector_expression<vec> &stockpath, int tindex)
    {
        const vec &v = stockpath;
        return immediate(v, tindex);
    }
private:
    /*! The strike price
     * 
     * put strike price
     */
    double K; ///< strike price
};



class asiancall : public payofffunc
{
public:
    asiancall(double K_) : K(K_) { }
    double immediate(const vec &stockpath, int tindex) 
    { 
        // for an asian, we need the average of stockpath from 0 to tindex
        double sum = 0.;
        for(int i = 0; i <= tindex; i++)
            sum += stockpath(i); 

        double avg = sum/(tindex+1.);
        return max(0., avg-K); 
    }
    double immediate(const vector_expression<vec> &stockpath, int tindex) 
    {
        const vec &v = stockpath;
        return immediate(v, tindex);
    }
private:
    double K;
};





// 
/*! \fn double calcstockprice(double x, double r, double sig, double T, double S0)
 * \brief calculates the stockprice at T from S0 given Gaussian x
 * 
 * \param x The Gaussian which we use to form the stock price
 * \param r risk-free rate
 * \param sig volatility
 * \param T the time at which we compute the stock price
 * \param S0 initial stock price
 * \return Lognormally distributed stock price
 */
double calcstockprice(double x, double r, double sig, double T, double S0)
{
    return S0*exp((r-sig*sig/2.)*T+sig*sqrt(T)*x);
}

/*! \fn mat &calcgaussians(gsl_rng *r0, mat &gaussians)
 * \brief Generates a matrix of standard unit Gaussians
 *
 * \param r0 GSL random number generator structure
 * \param gaussians the matrix to store the Gaussians in 
 * \return matrix of Gaussian mean=0 stdev=1
 */

mat &calcgaussians(gsl_rng *r0, mat &gaussians)
{
    for(int i = 0; i < (int)gaussians.size1(); i++)
    {
        for(int j = 0; j < (int)gaussians.size2(); j++)
        {
            gaussians(i, j) = gsl_ran_ugaussian(r0);
        }
    }
    return gaussians;
}

/*! \fn mat &calcstockprice(double S0, double sig, double r, double deltat, mat &gaussians, mat &stockprices)
  * \brief produce a matrix of Geometric Brownian Motions
  * with each row representing a stock price trajectories 
  *
  * \param S0 initial stock price
  * \param sig volatility
  * \param r risk-free rate
  * \param deltat timestep size
  * \param gaussians the matrix of Gaussians used to
  * generate GBM
  * \param stockprices the matrix where we store the GBMs
  * \return matrix of GBM stock price trajectories stored row-wise
  */
mat &calcstockprice(double S0, double sig, 
        double r, double deltat, mat &gaussians,
        mat &stockprices)
{
    for(int i = 0; i < (int)stockprices.size1(); i++)
    {
        stockprices(i, 0) = S0;
        for(int j = 1; j < (int)stockprices.size2(); j++)
        {
            stockprices(i, j) =
                calcstockprice(gaussians(i, j-1),
                        r, sig, deltat,
                        stockprices(i, j-1));
        }
    }
    return stockprices;
}
/*! \fn mat &applyjumpdiff(gsl_rng *r0, double lambda, double mu, double sig2, mat &stockprices, mat &jumpdiff)
  * \brief apply Poisson jump process to a matrix of
  * geometric Brownian motions
  *
  * \param r0 GSL random number generator
  * \param lambda jump arrival intensity
  * \param mu lognormal jump size mean
  * \param sig2 lognormal jump size scale
  * \param stockprices the matrix of the GBMs
  * \param jumpdiff where the stock price trajectories with jumps are
  * to be stored
  * \return matrix of stock price trajectories with Poisson jumps stored row-wise
  */

mat &applyjumpdiff(gsl_rng *r0, double lambda, double mu, double sig2, mat &stockprices, mat &jumpdiff)
{
    jumpdiff = stockprices;
    for(int i = 0; i < (int)jumpdiff.size1(); i++)
    {
        int rnext = int(ceil(gsl_ran_exponential(r0,1./lambda)));
        double ynow = 1.;
        for(int j = 0; j < (int)jumpdiff.size2(); j++)
        {
            if(rnext < j)
            {
                ynow *= gsl_ran_lognormal(r0, mu, sig2);
                rnext += int(ceil(gsl_ran_exponential(r0,1./lambda)));
            }
            jumpdiff(i, j) *= ynow;
        }

    }

    return jumpdiff;
}


double calceuroprice(payofffunc &option, double r, double deltat, mat &stockpaths) 
{
    double payoffsum = 0.;
    for(int i = 0; i < (int)stockpaths.size1(); i++)
    {
        const vec &stockpath = row(stockpaths, i);

        payoffsum += option.immediate(stockpath, stockpaths.size2()-1);
        cout << payoffsum << endl;
    }
    double optavg = payoffsum/stockpaths.size1();
    return exp(-r*deltat*stockpaths.size2())*optavg;
}

/*! \fn double calcamprice(payofffunc &option, double r, double deltat, mat &stockpaths) 
 * 
 * \brief Computes the LSM price of an American (Bermudan) option
 *
 * @param option the option payoff, depends on stockpath and t
 * @param r risk-free rate
 * @param deltat timestep size
 * @param stockpaths matrix of stock trajectories in rows, timesteps
 *          in columns
 * @return Bermudan approximation price of the American option
 *
 * The LSM works by starting at expiration, forming a continuation estimate by regressing
 * cashflows against underlying stock prices, and then updating cashflows if immediate exercise
 * is greater than continuation.  It proceeds from expiration down to t=0.
 * 
 * -# We begin the LSM at the expiration of the option.  We initialize the cashflow vectors to the
 *    European payoffs, since these have no uncertainty, we simply exercise if the option expires in
 *    the money.
 * -# We next loop over timesteps starting just before expiration, and going down to t=0.  At each
 *    timestep, we do the following steps:
 *    -# At each timestep, we count how many paths are in the money (ITM). (step 2.1)
 *    -# We initialize a basis matrix \f$A\f$ and a discounted cashflow vector \f$y\f$ for each ITM
 *       path.
 *    -# For each in-the-money path we do the following steps to prepare the basis matrix and
 *       discounted cashflow vector (step 2.2):
 *       -# Add a row to the basis matrix of \f$1, s, s^2\f$ where \f$s\f$ is the underlying stock
 *          price (step 2.2.1)
 *       -# Add the cashflows from the current timestep to expiration for the ITM path discounted back
 *          to the current time. (step 2.2.2)
 *    -# We next compute \f$A^\top A\f$ and  \f$A^\top y\f$, and then use LU-decomposition to solve
 *       the linear system \f$A^\top A c = A^\top y\f$ which causes \f$c=(A^\top A)^{-1} A^\top y\f$
 *       which is the solution for least-squares regression.  Hence we have regressed the discounted
 *       payoff vector against the underlying stock price, so we can predict continuation values from
 *       stock prices.  
 *    -# We form \f$C=A c\f$ which gives us the predicted continuation values at the stock prices
 *       of the ITM paths (step 2.3)
 *    -# We next compare these predicted continuation values with the immediate exercise values over
 *       each ITM path (step 2.4)
 *       -# If the regressed (predicted) continuation value is less than the immediate exercise, then
 *          we update the cashflow and cashflowtime vectors to reflect immediate exercise at the
 *          current timestep (step 2.4.2)
 *       -# We use inmoneypath to loop over ITM paths
 * -# Use the cashflow vectors to find the average cashflow over all the paths discounted at the
 *    exercise times.
 */


                 
double calcamprice(payofffunc &option, double r, double deltat, mat &stockpaths) 
{
    //cout << "step 1 initialize the cashflow vectors to european" << endl;
    const int Ti = stockpaths.size2()-1; //index of last timestep
    const double T = Ti*deltat; //chronological time of last timestep

    vec cashflow(stockpaths.size1()), cashflowtime(stockpaths.size1(),T);

    for(int i = 0; i < (int)stockpaths.size1(); i++)
        cashflow(i) = option.immediate(row(stockpaths, i), Ti);

    //cout << cashflow << endl << cashflowtime << endl;


    //cout << "step 2 ti =  setup loop ti=t-1 .. 0" << endl;
    for(int ti = Ti-1; ti >= 1; ti--)
    {
        //cout << "  step 2.1 find number of paths itm" << endl;
        int numinmoney = 0;

        for(int i = 0; i < (int)stockpaths.size1(); i++)
            if(option.immediate(row(stockpaths, i), ti) > 0.)
                numinmoney++;

        //cout << "ti = " << ti << "numinmoney=" << numinmoney << endl;

        mat basismat(numinmoney, 3);
        vec discopayoff(numinmoney);
        int inmoneypath = 0;
        //cout << "  step 2.2 for each itm path" << endl;
        for(int i = 0; i < (int)stockpaths.size1(); i++)
        {
            if(option.immediate(row(stockpaths, i), ti) > 0.)
            {
                //cout << "    step 2.2.1 make a basis mat of 1, s, s^2, e^{-s} for s=s_ti for each itm path" << endl;
                basismat(inmoneypath, 0) = 1.;
                basismat(inmoneypath, 1) = stockpaths(i, ti);
                basismat(inmoneypath, 2) = stockpaths(i, ti)*stockpaths(i, ti);
                //cout << "    step 2.2.2 make disco payoff vector" << endl;
                discopayoff(inmoneypath) =
                    cashflow(i)*exp(-r*(cashflowtime(i)-ti*deltat));
                inmoneypath++;
            }
        }
        //cout << "basis matrix = " << basismat << endl;
        //cout << "disco payoff= " << discopayoff << endl;

        mat ata = prod(trans(basismat), basismat); // A^t A
        vec aty = prod(trans(basismat), discopayoff); // A^t b

        permutation_matrix<size_t> pmat(ata.size1());

        const int factres = lu_factorize(ata, pmat);
        if(0 == factres)
            lu_substitute(ata, pmat, aty);
        else 
        {
            cerr << "warning error LSM Cannot LU Factorize for price" << endl;
            continue;
        }
        // aty should now hold the solution c matrix

        //cout << "  step 2.3 do regression of payoff vs basis mat" << endl;
        const vec c = aty;
        //cout << "c = " << c << endl;
        const vec contval = prod(basismat, c);
        //cout << "    step 2.4.1 find regressed continuation value" << endl;
        //cout << "contval = " << contval << endl;


        //cout << "  step 2.4 for each itm path" << endl;
        //cout << "    step 2.4.2 if continuation < immex, exercise ie update cashflow and cashflowtime" << endl;

        inmoneypath = 0;
        for(int i = 0; i < (int)stockpaths.size1(); i++)
        {
            double immex = option.immediate(row(stockpaths, i), ti);
            if(immex > 0.)
            {
                if(contval(inmoneypath) < immex)
                {
                    //cout << "EXERCISE bc " << contval(inmoneypath)
                     //   << " < " << immex << endl;
                    //update cashflow cashflowtime to exercise now
                    cashflow(i) = immex; // HW 
                    cashflowtime(i) = deltat*ti; // HW
                }
                inmoneypath++; // HW deal with inmoneypath
            }
        }


    }



    //cout << "step 3 average discounted cashflows" << endl;
    double cashsum = 0.;
    for(int i = 0; i < (int)cashflow.size(); i++)
        cashsum += exp(-r*deltat*cashflowtime(i))*cashflow(i);

    return cashsum/cashflow.size();
}

/*! \fn mat &applyjumps(gsl_rng *r0, double meanarrival, double jumpmean, double jumpsig, mat &stockprices, mat &jumpedprices)
 * \brief apply the Poisson jumps to the stock prices
 *
 * @param r0 GSL random number generator
 * @param meanarrival parameter for interarrival times from exponential distribution
 * @param jumpmean 

    
mat &applyjumps(gsl_rng *r0, double meanarrival, double jumpmean,
        double jumpsig, mat &stockprices, mat &jumpedprices)
{
    jumpedprices = stockprices;
    for(int i = 0; i < (int)jumpedprices.size1(); i++)
    {
        double jumptime = gsl_ran_exponential(r0, 1./meanarrival);
        double jumpsize = gsl_ran_lognormal(r0, jumpmean, jumpsig);

        double currentjumpsize = 1.;
        for(int t = 0; t < (int)jumpedprices.size2(); t++)
        {
            // this for loop lets us carry the jump to all the times
            // on path i 

            // i detect when t has passed the jump arrival and
            // then update currentjumpsize to apply the jumpsize to the
            // "jumpedprices" by doing:
            if(t > jumptime) 
            {
                currentjumpsize = 1.;
                    // HW how do i fix this to incorporate the
                    // jumpsize into currentjumpsize?
                    // notice you can have more than 1 jump occur
                    // before the expiration of the option.  
                //"gsl_ran_exponential(r0, 1./meanarrival);
            }

            // this multiplies all the later prices by the
            // currentjumpsize
            jumpedprices(i, t) *= currentjumpsize;
        }
    }
}


int main()
{
    const gsl_rng_type * T0;
    gsl_rng * r0;
    gsl_rng_env_setup();
    T0 = gsl_rng_default;
    r0 = gsl_rng_alloc (T0);

    int n = 50, numpaths = 500;
    double optionsum = 0.;
    double r = .06, S0 = 100., sig=.2, K=100.;
    double deltat = 1.;
    double deltath = deltat/100.;

    mat gaussians(numpaths, n);
    calcgaussians(r0, gaussians);

    mat stockprices(numpaths, n+1), bumpup(numpaths,n+1), bumpdown(numpaths,n+1);
    mat smallsteps(numpaths, n+1), bigsteps(numpaths, n+1);
    mat morevega(numpaths, n+1), lessvega(numpaths, n+1);
    double h = 1.;
    // generate the stock paths from the SAME set of gaussians when hedging

    calcstockprice(S0, sig, r, deltat, gaussians, stockprices);
    double meanarrival = 10., jumpmean = 1., jumpsig = .5;
    mat jumpedprices;
    applyjumps(r0, meanarrival, jumpmean, jumpsig, stockprices,
            jumpedprices);



    double sigh = sig/100.;
    double hisig=sig+sigh, losig=sig-sigh;
    //bumped prices for hedging
    calcstockprice(S0+h, sig, r, deltat, gaussians, bumpup);
    calcstockprice(S0-h, sig, r, deltat, gaussians, bumpdown);
    calcstockprice(S0, sig, r, deltat-deltath, gaussians, smallsteps);
    calcstockprice(S0, sig, r, deltat+deltath, gaussians, bigsteps);
    calcstockprice(S0, hisig, r, deltat, gaussians, morevega);
    calcstockprice(S0, losig, r, deltat, gaussians, lessvega);
    
    putpayoff atm(S0), dim(S0/2.), ffm(2.*S0);

    mat le(8, 4);
    le(0,0)=1.; le(0,1)=1.09; le(0,2)=1.08; le(0,3)=1.34;
    le(1,0)=1.; le(1,1)=1.16; le(1,2)=1.26; le(1,3)=1.54;
    le(2,0)=1.; le(2,1)=1.22; le(2,2)=1.07; le(2,3)=1.03;
    le(3,0)=1.; le(3,1)=0.93; le(3,2)=0.97; le(3,3)=0.92;
    le(4,0)=1.; le(4,1)=1.11; le(4,2)=1.56; le(4,3)=1.52;
    le(5,0)=1.; le(5,1)=0.76; le(5,2)=0.77; le(5,3)=0.9;
    le(6,0)=1.; le(6,1)=0.92; le(6,2)=0.84; le(6,3)=1.01;
    le(7,0)=1.; le(7,1)=0.88; le(7,2)=1.22; le(7,3)=1.34;
    putpayoff lsmput(1.1);
    asiancall ac(K);

    //cout << calcamprice(lsmput, r, deltat, le) << endl;
    //cout << calcamprice(ac, r, deltat, le) << endl;

    double c0 = calcamprice(ac, r, deltat, stockprices);
    double cup = calcamprice(ac, r, deltat, bumpup);
    double cdown = calcamprice(ac, r, deltat, bumpdown);
    double csmallsteps = calcamprice(ac, r, deltat-deltath, smallsteps);
    double cbigsteps = calcamprice(ac, r, deltat+deltath, bigsteps);
    double chisig = calcamprice(ac, r, deltat, morevega);
    double closig = calcamprice(ac, r, deltat, lessvega);

    cout << "asian call s0  =" << c0 << endl;
    cout << "asian call s0+h=" << cup << endl; 
    cout << "asian call s0-h=" << cdown << endl; 
    cout << "asian call delt=" << (cup-cdown)/(2.*h) << endl;
    cout << "asian call gamm=" << (cup-2.*c0+cdown)/(h*h) << endl;
    cout << "asian call tau =" << (cbigsteps-csmallsteps)/(2.*deltath*n) << endl;
    cout << "asian call bern=" << (cbigsteps-2.*c0+csmallsteps)/(deltath*deltath*n*n) << endl;
    cout << "chisig =  " << chisig << endl;
    cout << "closig =  " << closig << endl;
    
    cout << "asian call vega=" << (chisig-closig)/(2.*sigh) << endl;
    cout << "asian call v2=" << (chisig-2.*c0+closig)/(sigh*sigh) << endl;

    gsl_rng_free (r0);

    return 0;
}
    //mat jumped;
    //applyjumpdiff(r0, .1, 0., .1, stockprices, jumped);
