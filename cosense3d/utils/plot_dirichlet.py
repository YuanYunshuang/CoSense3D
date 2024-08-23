import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns
from math import gamma
from operator import mul
from functools import reduce

sns.set(style='white', font_scale=1.2)
sns.color_palette("Blues", as_cmap=True)


def plot_mesh(corners):
    """Subdivide the triangle into a triangular mesh and plot the original and subdivided triangles."""
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)

    plt.figure(figsize=(6, 4))
    for i, mesh in enumerate((triangle, trimesh)):
        plt.subplot(1, 2, i + 1)
        plt.triplot(mesh)
        plt.axis('off')
        plt.axis('equal')


class Dirichlet:
    """Define the Dirichlet distribution with vector parameter alpha."""

    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / reduce(mul, [gamma(a) for a in self._alpha])

    def pdf(self, x):
        """Returns pdf value for `x`. """
        return self._coef * reduce(mul, [xx ** (aa - 1) for (xx, aa) in zip(x, self._alpha)])


class PlotDirichlet:
    """
    Plot the Dirichlet distribution as a contour plot on a 2-Simplex.
    """

    def __init__(self, corners):
        self._corners = corners
        self._triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
        # Midpoints of triangle sides opposite of each corner
        self._midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0 for i in range(3)]

    def xy2bc(self, xy, tol=1.e-3):
        """Map the x-y coordinates of the mesh vertices to the simplex coordinate space (aka barycentric coordinates).
        Here we use a simple method that uses vector algebra. For some values of alpha, calculation of the Dirichlet pdf
        can become numerically unstable at the boundaries of the simplex so our conversion function will take an optional
        tolerance that will avoid barycentric coordinate values directly on the simplex boundary.
        """
        s = [(self._corners[i] - self._midpoints[i]).dot(xy - self._midpoints[i]) / 0.75 for i in range(3)]
        return np.clip(s, tol, 1.0 - tol)

    def draw_pdf_contours(self, ax, dist, label=None, nlevels=50, subdiv=8, **kwargs):
        """Draw pdf contours for a Dirichlet distribution"""
        # Subdivide the triangle into a triangular mesh
        refiner = tri.UniformTriRefiner(self._triangle)
        trimesh = refiner.refine_triangulation(subdiv=subdiv)

        # convert to barycentric coordinates and compute probabilities of the given distribution
        pvals = [dist.pdf(self.xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]

        ax.tricontour(trimesh, pvals, 10, linewidths=0.1, colors="gray")
        tcf = ax.tricontourf(trimesh, pvals, nlevels, cmap="jet", **kwargs)
        ax.plot([0, 0.5], [0, 0.75 ** 0.5], 'k')
        ax.plot([0, 1], [0, 0])
        ax.plot([0.5, 1], [0.75 ** 0.5 , 0], 'k')
        # plt.axis('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.75 ** 0.5)
        ax.set_title("$\\alpha$ = " + str(label))
        ax.axis('off')

        return ax, tcf


if __name__ == '__main__':
    kwargs = {}
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75 ** 0.5]])
    plot_dirichlet = PlotDirichlet(corners)

    f, axes = plt.subplots(2, 3, figsize=(14, 6))
    ax = axes[0, 0]
    alpha = (0.9, 0.9, 0.9)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%.1f")

    ax = axes[0, 1]
    alpha = (1, 1, 1)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%2.1f")

    ax = axes[0, 2]
    alpha = (3, 3, 3)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%2.1f")

    ax = axes[1, 0]
    alpha = (1, 3, 3)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%2.1f")

    ax = axes[1, 1]
    alpha = (2, 2, 10)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%2.1f")

    ax = axes[1, 2]
    alpha = (20, 20, 20)
    dist = Dirichlet(alpha)
    ax, tcf = plot_dirichlet.draw_pdf_contours(ax, dist, alpha, **kwargs)
    f.colorbar(tcf, ax=ax, format="%2.1f")

    f.savefig('/home/yys/Downloads/Dirichlet.pdf', bbox_inches='tight', transparent=True)