/*
TODO: Description of modifications introduced and my copyright.
*/

/*
Copyright University of Amsterdam, 2002-2004. All rights reserved.

Contact person:
Jan-Mark Geusebroek (mark@science.uva.nl, http://www.science.uva.nl/~mark)
Intelligent Systems Lab Amsterdam
Informatics Institute, Faculty of Science, University of Amsterdam
Kruislaan 403, 1098 SJ Amsterdam, The Netherlands.


This software is being made available for individual research use only.
Any commercial use or redistribution of this software requires a license from
the University of Amsterdam.

You may use this work subject to the following conditions:

1. This work is provided "as is" by the copyright holder, with
absolutely no warranties of correctness, fitness, intellectual property
ownership, or anything else whatsoever.  You use the work
entirely at your own risk.  The copyright holder will not be liable for
any legal damages whatsoever connected with the use of this work.

2. The copyright holder retain all copyright to the work. All copies of
the work and all works derived from it must contain (1) this copyright
notice, and (2) additional notices describing the content, dates and
copyright holder of modifications or additions made to the work, if
any, including distribution and use conditions and intellectual property
claims.  Derived works must be clearly distinguished from the original
work, both by name and by the prominent inclusion of explicit
descriptions of overlaps and differences.

3. The names and trademarks of the copyright holder may not be used in
advertising or publicity related to this work without specific prior
written permission. 

4. In return for the free use of this work, you are requested, but not
legally required, to do the following:

- If you become aware of factors that may significantly affect other
	users of the work, for example major bugs or
	deficiencies or possible intellectual property issues, you are
	requested to report them to the copyright holder, if possible
	including redistributable fixes or workarounds.

- If you use the work in scientific research or as part of a larger
	software system, you are requested to cite the use in any related
	publications or technical documentation. The work is based upon:

		J. M. Geusebroek, A. W. M. Smeulders, and J. van de Weijer.
		Fast anisotropic gauss filtering. IEEE Trans. Image Processing,
		vol. 12, no. 8, pp. 938-943, 2003.
 
	related work:

		I.T. Young and L.J. van Vliet. Recursive implementation
		of the Gaussian filter. Signal Processing, vol. 44, pp. 139-151, 1995.
 
		B. Triggs and M. Sdika. Boundary conditions for Young-van Vliet
		recursive filtering. IEEE Trans. Signal Processing,
		vol. 54, pp. 2365-2367, 2006.
 
This copyright notice must be retained with all copies of the software,
including any modified or derived versions.
*/

#ifndef ANIGAUSS_H
#define ANIGAUSS_H

#include <stdlib.h>
#include <math.h>


/* define the input buffer type, e.g. "float" */
#define SRCTYPE double

/* define the output buffer type, should be at least "float" */
#define DSTTYPE double

/*
 *  the main function:
 *    anigauss(inbuf, outbuf, bufwidth, bufheight, sigma_v, sigma_u, phi,
 *       derivative_order_v, derivative_order_u);
 *
 *  v-axis = short axis
 *  u-axis = long axis
 *  phi = orientation angle in degrees
 *
 *  for example, anisotropic data smoothing:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 *  or, anisotropic edge detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 1, 0);
 *
 *  or, anisotropic line detection:
 *    anigauss(inptr, outptr, 512, 512, 3.0, 7.0, 30.0, 2, 0);
 *
 *  or, in-place anisotropic data smoothing:
 *    anigauss(bufptr, bufptr, 512, 512, 3.0, 7.0, 30.0, 0, 0);
 *
 */

/* the function prototypes */
int anigauss(SRCTYPE *input, DSTTYPE *output, int sizex, int sizey,
	double sigmav, double sigmau, double phi, int orderv, int orderu);


#endif