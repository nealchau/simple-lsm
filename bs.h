#ifndef BS_HH  // include guard
#define BS_HH

// g++ will say it's an error if you define the same function 
// twice.  so if this file gets #included twice without the 
// include guards, it will cause
// an error.  so we put "#include guards" to prevent our functions
// from being defined multiply.

double d1(double r, double T, double sig, double S0, double K);
double d2(double r, double T, double sig, double S0, double K);
double N(double x);
double bscall(double r, double T, double sig, double S0, double K);

#endif
