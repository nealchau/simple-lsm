#include <cmath>
#include <iostream>

using namespace std;

#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

double d1(double r, double T, double sig, double S0, double K)
{
	double numerator = log(S0/K)+(r+sig*sig/2.)*T;
	return numerator/(sig*sqrt(T));
}

double d2(double r, double T, double sig, double S0, double K)
{	
	return d1(r, T, sig, S0, K) - sig*sqrt(T);
}

double N(double x)
{
	return gsl_cdf_ugaussian_P(x);
}


double bscall(double r, double T, double sig, double S0, double K)
{
	double d1val = d1(r, T, sig, S0, K);
	double d2val = d2(r, T, sig, S0, K);


	return S0*N(d1val)-K*exp(-r*T)*N(d2val);

	//return S0*N(d1(S0, K, r, sig, T)) 
	//	-K*exp(-r*T)*N(d2(S0, K, r, sig, T));

}
double bsput(double r, double T, double sig, double S0, double K)
{
	double d1val = d1(r, T, sig, S0, K);
	double d2val = d2(r, T, sig, S0, K);


	return K*exp(-r*T)*N(-d2val)-S0*N(-d1val);
}



/* this main function is now a comment
main()
{
	double x;
	
	cout << gsl_cdf_ugaussian_P(x) << endl;
	cout << N(x) << endl;
	cout << "d1=" << d1(.05, 1., .1, 1., 1.) << endl;
	cout << "d2=" << d2(.05, 1., .1, 1., 1.) << endl;
	cout << "bscall=" << bscall(.05, 1., .1, 1., 1.) << endl;

	return 0;
}
*/
