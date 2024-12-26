# **Contributing to Totsu**

Thank you for considering contributing to Totsu! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

---

## **How to Contribute**

### **1. Reporting Issues**
If you find a bug or have a feature request, please:
1. **Search existing issues** to ensure your issue hasn’t already been reported.
2. If it hasn’t, open a new issue and provide:
   - A clear and descriptive title.
   - Steps to reproduce the issue.
   - Any relevant error messages or screenshots.

---

### **2. Suggesting Enhancements**
We welcome ideas for improving Totsu! To suggest enhancements:
1. Open an issue with the label `enhancement`.
2. Describe your idea clearly, including why it would be useful.

---

### **3. Submitting Code Changes**
To make a code contribution:
1. **Fork this repository** and create your own branch:
   ```bash
   git checkout -b feature/YourFeatureName
   ```
2. **Make your changes** with clear commit messages.
3. Test your changes:
   - Run `pytest` to ensure all tests pass.
   - Add tests if needed for new features.
4. **Submit a pull request**:
   - Explain what the pull request does.
   - Reference any related issues or discussions.

---

## **Development Setup**

### **Prerequisites**
Make sure you have the following installed:
- Python 3.8+ 
- Conda (recommended for environment management)

### **Setup Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/ggsato/totsu.git
   cd totsu
   ```
2. Create a Conda environment:
   ```bash
   conda create -n totsu-env python=3.8
   conda activate totsu-env
   ```
3. Install dependencies:
   ```bash
   conda install -c conda-forge pyomo numpy pytest plotly dash dash-bootstrap-components
   ```
4. Run tests to ensure the setup works:
   ```bash
   pytest
   ```

---

## **Code of Conduct**
By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please ensure that all interactions are respectful and constructive.

---

## **Getting Help**
If you need assistance or have any questions about contributing, feel free to:
- Open a GitHub issue.
- Join the community discussion (Not available, yet. Open an issue!).

---

Thank you for helping make Totsu better! 🚀