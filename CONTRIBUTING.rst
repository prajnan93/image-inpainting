
Get Started!
------------

Here's how to set up `inpaint` for local development.

1. Install [Miniconda](https://conda.io/miniconda.html)

2. Clone the repo locally::

    $ git clone https://github.com/prajnan93/image-inpainting

3. Create a Conda virtual environment using the `environment.yml` file. Install your local copy of the package into the environment::

    $ conda env create -f environment.yml
    $ conda activate inpaint
    $ pre-commit install
    $ python setup.py develop
    
4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. Ensure you write tests for the code you add and run the tests before you commit. You can run tests locally using `pytest` from the root directory of the repository::

    $ pytest

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

    Note: You might need to add and commit twice if pre-commit hooks modify your code.

7. Submit a pull request through the GitHub website.