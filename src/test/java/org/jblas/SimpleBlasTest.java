// --- BEGIN LICENSE BLOCK ---
/* 
 * Copyright (c) 2009, Mikio L. Braun
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 *     * Neither the name of the Technische Universität Berlin nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// --- END LICENSE BLOCK ---

package org.jblas;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Some test for class SimpleBlas
 *
 * @author Mikio L. Braun
 */
public class SimpleBlasTest {

    private final double eps = 1e-10;

    @Test
    public void testGeev() {
        DoubleMatrix A = new DoubleMatrix(2, 2, 3.0, -3.0, 1.0, 1.0);
        DoubleMatrix WR = new DoubleMatrix(2);
        DoubleMatrix WI = new DoubleMatrix(2);
        DoubleMatrix VR = new DoubleMatrix(2,2);
        DoubleMatrix VL = new DoubleMatrix(2,2);
        
        SimpleBlas.geev('V', 'N', A, WR, WI, VR, VL);
        
        assertTrue(new DoubleMatrix(2, 1, 2.0, 2.0).compare(WR, 1e-6));
        assertTrue(new DoubleMatrix(2, 1, Math.sqrt(2.0), -Math.sqrt(2.0)).compare(WI, 1e-6));
        
        /*System.out.printf("WR = %s\n", WR.toString());
        System.out.printf("WI = %s\n", WI.toString());
        System.out.printf("VR = %s\n", VR.toString());
        System.out.printf("VL = %s\n", VL.toString());
        System.out.printf("A = %s\n", A.toString());*/
    }

    @Test
    public void testSwapReal() {
        DoubleMatrix x = new DoubleMatrix(new double[]{1, 2, 3});
        DoubleMatrix y = x.dup().mul(-1);

        SimpleBlas.swap(x, y);

        assertEquals(-6, x.sum(), eps);
        assertEquals(6, y.sum(), eps);

    }

    @Test
    public void testSwapComplex() {
        ComplexDoubleMatrix x = new ComplexDoubleMatrix(
                new DoubleMatrix(new double[][]{{1, 1}, {1, 1}}),
                new DoubleMatrix(new double[][]{{0, 0}, {0, 0}})
        );
        ComplexDoubleMatrix y = new ComplexDoubleMatrix(
                new DoubleMatrix(new double[][]{{0, 0}, {0, 0}}),
                new DoubleMatrix(new double[][]{{1, 1}, {1, 1}})
        );

        SimpleBlas.swap(x, y);

        assertEquals(4, x.sum().imag(), eps);
        assertEquals(4, y.sum().real(), eps);
    }

    @Test
    public void testGemv() {
        DoubleMatrix A = new DoubleMatrix(new double[][]{{1.0, 4.0, 7.0}, {2.0, 5.0, 8.0}, {3.0, 6.0, 9.0}});
        DoubleMatrix x = new DoubleMatrix(new double[]{1.0, 3.0, 7.0});
        DoubleMatrix y = new DoubleMatrix(new double[]{0.0, 0.0, 0.0});

        DoubleMatrix y1 = SimpleBlas.gemv(1.0, A, x, 0.0, y.dup());
        ComplexDoubleMatrix y2 = SimpleBlas.gemv(new ComplexDouble(1.0), A.toComplex(), x.toComplex(),
                new ComplexDouble(0.0), y.toComplex().dup());

        assertArrayEquals(new double[]{62.0, 73.0, 84.0}, y1.data, eps);
        assertArrayEquals(new double[]{62.0, 73.0, 84.0}, y2.real().data, eps);
    }
}
