CC = g++
CXXFLAGS = -Wall  -pg -ggdb
LDFLAGS = -ggdb
LDLIBS = -lm -lgsl -lgslcblas -lstdc++ 
CTAGS=ctags

sources = mcbm.C bs.C mcbmc.C

all: mcbm mcbmc tags
clean:
	rm -f *.o *.d

mcbm: mcbm.o bs.o
mcbmc: mcbmc.o bs.o
tags:   ${sources}
	${CTAGS} ${sources}

include $(sources:.C=.d)

%.d: %.C
	@set -e; rm -f $@; \
	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

# vim:noexpandtab nosmarttab:
