import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

class EllipticCurve:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.discriminant = 4 * a**3 + 27 * b**2
        if self.discriminant == 0:
            print("Warning: The curve is singular.")

    def plot_real(self):
        """Plots the curve over real numbers."""
        print(f"Plotting y^2 = x^3 + {self.a}x + {self.b} over real numbers.")

        x = np.linspace(-5, 5, 400)
        y_squared = x**3 + self.a*x + self.b
        y = np.sqrt(np.maximum(y_squared, 0))

        plt.figure(figsize=(10, 8))
        plt.plot(x, y, 'b', label=f'$y^2 = x^3 + {self.a}x + {self.b}$')
        plt.plot(x, -y, 'b')
        plt.title('Elliptic Curve Plot (Real Numbers)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

class FiniteFieldEllipticCurve(EllipticCurve):
    def __init__(self, p, a, b):
        super().__init__(a, b)
        self.p = p
        if self.p <= 1:
            raise ValueError("p must be a prime number greater than 1.")
        if (4 * a**3 + 27 * b**2) % p == 0:
            print("Warning: The curve is singular over Fp.")

        self.points = self._calculate_points()

    def _calculate_points(self):
        """Calculates all points on the curve over Fp."""
        # Pre-calculate quadratic residues
        squares = {pow(y, 2, self.p): [] for y in range(self.p)}
        for y in range(self.p):
            squares[pow(y, 2, self.p)].append(y)

        points = []
        for x in range(self.p):
            rhs = (x**3 + self.a * x + self.b) % self.p
            if rhs in squares:
                for y in squares[rhs]:
                    points.append((x, y))

        points.append(None) # Point at infinity
        return points

    def list_points(self):
        """Prints all the points on the curve."""
        print(f"\nPoints on the curve y^2 = x^3 + {self.a}x + {self.b} (mod {self.p}):")
        print("--------------------------------------------------")
        for point in self.points:
            print(f" {point}")
        print(f"\nTotal number of points (including the point at infinity): {len(self.points)}")

    def add_points(self, p1, p2):
        """Adds two points on the curve."""
        if p1 is None:
            return p2
        if p2 is None:
            return p1

        x1, y1 = p1
        x2, y2 = p2

        if x1 == x2 and y1 != y2:
            return None # p1 + (-p1) = O

        if x1 == x2: # Point doubling
            if y1 == 0:
                return None # Tangent is vertical
            m = (3 * x1**2 + self.a) * pow(2 * y1, -1, self.p)
        else: # Point addition
            m = (y2 - y1) * pow(x2 - x1, -1, self.p)

        x3 = (m**2 - x1 - x2) % self.p
        y3 = (m * (x1 - x3) - y1) % self.p
        return (x3, y3)

    def plot_finite_field(self):
        """Creates a scatter plot of the points on the curve over Fp."""
        plt.figure(figsize=(10, 10))

        x_coords = [p[0] for p in self.points if p is not None]
        y_coords = [p[1] for p in self.points if p is not None]

        plt.scatter(x_coords, y_coords)

        plt.title(f'Points on y^2 = x^3 + {self.a}x + {self.b} (mod {self.p})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks(range(self.p))
        plt.yticks(range(self.p))
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import sys
    if 'ipykernel' in sys.modules:
        print("Running in a notebook environment. Use the classes directly.")
        print("Example Usage:")
        print("\n# To plot a curve over real numbers:")
        print("# curve = EllipticCurve(a=-2, b=4)")
        print("# curve.plot_real()")
        print("\n# To list points on a curve over a finite field:")
        print("# ff_curve = FiniteFieldEllipticCurve(p=23, a=1, b=1)")
        print("# ff_curve.list_points()")
        print("\n# To plot points on a curve over a finite field:")
        print("# ff_curve.plot_finite_field()")
        print("\n# To add two points:")
        print("# p1 = (3, 10)")
        print("# p2 = (9, 7)")
        print("# ff_curve = FiniteFieldEllipticCurve(p=23, a=1, b=1) # Recreate curve if needed")
        print("# result = ff_curve.add_points(p1, p2)")
        print("# print(f'Result of {p1} + {p2} = {result}')")
    else:
        parser = argparse.ArgumentParser(description="Elliptic Curve Plotter and Point Calculator.")
        subparsers = parser.add_subparsers(dest='command', required=True)

        # Subparser for plotting over real numbers
        parser_plot = subparsers.add_parser('plot', help='Plot an elliptic curve over real numbers.')
        parser_plot.add_argument('-a', type=int, default=-2, help='Coefficient a of the curve.')
        parser_plot.add_argument('-b', type=int, default=4, help='Coefficient b of the curve.')

        # Subparser for finite field calculations
        parser_ff = subparsers.add_parser('finite_field', help='Calculate points on an elliptic curve over a finite field.')
        parser_ff.add_argument('-p', type=int, default=23, help='The prime modulus of the finite field.')
        parser_ff.add_argument('-a', type=int, default=1, help='Coefficient a of the curve.')
        parser_ff.add_argument('-b', type=int, default=1, help='Coefficient b of the curve.')
        parser_ff.add_argument('--plot', action='store_true', help='Plot the points on the curve.')

        # Subparser for point addition
        parser_add = subparsers.add_parser('add', help='Add two points on an elliptic curve over a finite field.')
        parser_add.add_argument('-p', type=int, required=True, help='The prime modulus of the finite field.')
        parser_add.add_argument('-a', type=int, required=True, help='Coefficient a of the curve.')
        parser_add.add_argument('-b', type=int, required=True, help='Coefficient b of the curve.')
        parser_add.add_argument('p1', type=str, help="First point, e.g., '(x,y)' or 'O'.")
        parser_add.add_argument('p2', type=str, help="Second point, e.g., '(x,y)' or 'O'.")

        args = parser.parse_args()

        if args.command == 'plot':
            curve = EllipticCurve(a=args.a, b=args.b)
            curve.plot_real()
        elif args.command == 'finite_field':
            ff_curve = FiniteFieldEllipticCurve(p=args.p, a=args.a, b=args.b)
            ff_curve.list_points()
            if args.plot:
                ff_curve.plot_finite_field()
        elif args.command == 'add':
            ff_curve = FiniteFieldEllipticCurve(p=args.p, a=args.a, b=args.b)

            def parse_point(p_str):
                if p_str.upper() == 'O':
                    return None
                try:
                    return tuple(map(int, p_str.strip('()').split(',')))
                except:
                    raise argparse.ArgumentTypeError(f"Invalid point format: {p_str}")

            p1 = parse_point(args.p1)
            p2 = parse_point(args.p2)

            result = ff_curve.add_points(p1, p2)

            p1_str = "O" if p1 is None else str(p1)
            p2_str = "O" if p2 is None else str(p2)
            res_str = "O" if result is None else str(result)

            print(f"Adding {p1_str} and {p2_str} on y^2 = x^3 + {args.a}x + {args.b} (mod {args.p})")
            print(f"Result: {res_str}")
