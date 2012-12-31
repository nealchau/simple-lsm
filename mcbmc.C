#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "bs.h"

using namespace std;
using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::vector<double> vec;
typedef boost::numeric::ublas::matrix<double> mat;

class payofffunc
{
    // this is a pure virtual base class. you derive classes (ie option payoffs) from it.
public:
    virtual double immediate(const vector_expression<vec> &stockpath, int tindex) = 0;
    virtual double immediate(const vec &stockpath, int tindex) = 0;
    // this means any class you derive from payofffunc must specify what the immediate exercise value is,
    // and for any object of class payofffunc, you can find the immediate exercise value.
};

class callpayoff : public payofffunc // callpayoff derived from payofffunc
{
public:
    callpayoff(double K_) : K(K_) { } // constructor of callpayoff.  
    // a constructor is a member function that has
    // the same name as the class its in, and it prepares the member data.  
    // in this case, the constructor stores the strike price in a data member.
    double immediate(const vec &stockpath, int tindex)
    {
        // this is the function which computes the immediate exercise value of a call as "promised" by our
        // pure virtual base class payofffunc
        return max(0., stockpath(tindex)-K);
    }
    double immediate(const vector_expression<vec> &stockpath, int tindex)
    {
        // this is the function which computes the immediate exercise value of a call as "promised" by our
        // pure virtual base class payofffunc
        const vec &v = stockpath;
        return immediate(v, tindex);
    }
private:
    double K;
};


// calculates final stock price at T from S0 and gaussian x
double calcstockprice(double x, double r, double sig, double T, double S0)
{
    return S0*exp((r-sig*sig/2.)*T+sig*sqrt(T)*x);
}

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
                 
double calcamprice(payofffunc &option, double r, double deltat, mat &stockpaths) 
{
    cout << "step 1 initialize the cashflow vectors to european" << endl;
    const int Ti = stockpaths.size2()-1;
    const double T = Ti*deltat;
    vec cashflow(stockpaths.size1()), cashflowtime(stockpaths.size1(), T);
    for(int i = 0; i < (int)stockpaths.size1(); i++)
        cashflow(i) = option.immediate(row(stockpaths, i), Ti);
    cout << cashflow << endl << cashflowtime <<endl;

    for(int ti = Ti-1; ti >= 0; ti--)
    {
        cout << "step 2 ti = " << ti << " setup loop ti=t-1 .. 0" << endl;
        cout << "  step 2.1 find number of paths itm" << endl;
        int numitm = 0;
        for(int i = 0; i < (int)stockpaths.size1(); i++)
            if(option.immediate(row(stockpaths, i), ti) > 0.) 
                numitm++;

        const int numbasisfn = 3;
        mat basis(numitm, numbasisfn, 0.);
        vec disco(numitm, 0.);
        int inmoneypath = 0;

        for(int i = 0; i < (int)stockpaths.size1(); i++)
            if(option.immediate(row(stockpaths, i), ti) > 0.)
            {
                basis(inmoneypath, 0) = 1.;
                basis(inmoneypath, 1) = stockpaths(i, ti);
                basis(inmoneypath, 2) = stockpaths(i, ti) * stockpaths(i, ti);
                disco(inmoneypath) = exp(r*(-cashflowtime(i)+ti*deltat))*cashflow(i);
                inmoneypath++;
            }


        cout << "  step 2.2 for each itm path" << endl;
        cout << "    step 2.2.1 make a basis mat of 1, s, s^2, e^{-s} for s=s_ti for each itm path" << endl;
        cout << "    step 2.2.2 make disco payoff vector" << endl;
        cout << "  step 2.3 do regression of payoff vs basis mat" << endl;
        cout << "  step 2.4 for each itm path" << endl;
        cout << "    step 2.4.1 find regressed continuation value" << endl;
        cout << "    step 2.4.2 if continuation < immex, exercise ie update cashflow and cashflowtime" << endl;
        cout << "step 3 average discounted cashflows" << endl;
    }

        return 0.;
    }

                


    int main()
    {
        const gsl_rng_type * T0;
        gsl_rng * r0;
        gsl_rng_env_setup();
        T0 = gsl_rng_default;
        r0 = gsl_rng_alloc (T0);

        int i, n = 10;
        double optionsum = 0.;
        double r = .01, T = .5, S0 = 100., sig=.2, K=100.;
        double deltat = .1;

        for (i = 0; i < n; i++) 
        {
            // gsl_ran_ugaussian generates random gaussians
            // which we can use to simulate stock prices.
            double g = gsl_ran_ugaussian(r0);
            cout << "gaussian=" << g;
            double sT = calcstockprice(g, r, sig, T, S0);
            cout << " stock price=" << sT;
            double callprice = max(0., sT-K);
            cout << " call price=" << callprice; 
            optionsum += callprice;
            cout << " optionsum=" << optionsum << endl; 
        }
        cout << "OPTION PRICE = " << exp(-r*T)*(optionsum/(double(n))) << endl;
        cout << "BS price = " << bscall(r, T, sig, S0, K) << endl;

        mat gaussians(10, 5);
        calcgaussians(r0, gaussians);
        cout << gaussians << endl;


        mat stockprices(10, 6);
        calcstockprice(S0, sig, r, deltat, gaussians, stockprices);
        cout << stockprices << endl;
        
        callpayoff atm(S0), dim(S0/2.), ffm(2.*S0);
        const vec &stockpath = row(stockprices, 2);
        cout << atm.immediate(stockpath, 3) << endl;
        cout << dim.immediate(stockpath, 3) << endl;
        cout << ffm.immediate(stockpath, 3) << endl;

        //binpayoff batmcall(S0); 
        cout << calceuroprice(atm, r, deltat, stockprices) << endl;
        cout << calceuroprice(dim, r, deltat, stockprices) << endl;
        cout << calcamprice(dim, r, deltat, stockprices) << endl;

        gsl_rng_free (r0);

        return 0;
    }
