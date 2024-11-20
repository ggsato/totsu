# Project Summary

*totsu* (凸), meaning "convex" in Japanese, is a project that reimagines the Simplex method and its integration into modern optimization workflows with a focus on educational clarity, robust design, and innovative visualization. The core goal of *totsu* is to provide a tool that not only solves linear programming problems efficiently but also demystifies the process, empowering users to explore, understand, and interact with convex optimization concepts.

The project introduces a modular and extensible Simplex solver built around a `Tableau` class that encapsulates the mechanics of the Simplex algorithm while maintaining an intuitive interface. Advanced visualization techniques, inspired by duality theory, provide users with interactive tools to explore feasible regions, objective progressions, and sensitivity analysis in the dual space. By integrating seamlessly with Pyomo and other solvers, *totsu* ensures consistency and reliability in complex workflows, making it an ideal tool for both practitioners and learners.

# Introduction

Optimization lies at the heart of decision-making, and linear programming is one of its most fundamental techniques. The Simplex method, celebrated for its efficiency and elegance, has been a cornerstone of linear programming for decades. However, the method's inner workings—tableau manipulations, pivot operations, and duality principles—can often feel opaque, especially for newcomers. *totsu* (凸) is a project that seeks to bridge this gap by combining the rigor of computational optimization with intuitive, visually-driven insights.

At the heart of *totsu* is a meticulously designed `Tableau` class, serving as the backbone of the Simplex algorithm. It provides robust tools for managing basis variables, pivot operations, and tableau state tracking. Complementing this is a Helper class that simplifies model standardization, ensuring that even non-standard problems can be seamlessly converted into a form ready for Simplex.

Where *totsu* truly shines is in its focus on visualization. By leveraging the dual model, it offers users a clear and interactive window into the optimization process. Visualizing dual variables and their direct impact on the objective value creates a straightforward, engaging experience. Coupled with interactive exploration tools, guided tutorials, and sensitivity analysis, *totsu* transforms the Simplex method into an accessible and enlightening journey.

Designed for integration with Pyomo, *totsu* ensures consistency in workflows involving Branch and Bound or dynamic constraint modifications. With its emphasis on educational value, modularity, and innovative design, *totsu* is more than a solver—it's a platform for learning, exploring, and advancing the understanding of convex optimization.
