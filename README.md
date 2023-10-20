<!-- PROJECT LOGO -->
<br />
<p align="center">
   <img src="https://github.com/martindoff/DC-RNN/RNN0.eps" alt="Logo" width="400" height="400">
  <p align="center">
   Recurrent Neural Network (RNN) model with difference-of-convex-functions (DC) structure. 
    <br />  
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Running the code</a></li>
        <li><a href="#options">Options</a></li>
      </ul>
    </li>
    <li><a href="#application">Application</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Learn dynamical systems from data using Recurrent Neural Network (RNN) models with a special 
difference-of-convex-functions (DC) structure. The so-called DC-RNN model approximates the
system trajectories by a DC input-output map between the control input sequence and state trajectory.
The DC-RNN model is leveraged in this example to predict the trajectory of the  
[coupled tank](https://ora.ox.ac.uk/objects/uuid:a3a0130b-5387-44b3-97ae-1c9795b91a42/download_file?safe_filename=Doff-Sotta_and_Cannon_2022_Difference_of_convex.pdf&file_format=application%2Fpdf&type_of_work=Conference+item)
dynamical system.  

### Built With

* Python 3
* Keras



<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

You need to install the following:
* numpy
* matplotlib
* [tensorflow / keras](https://keras.io/getting_started/)

Run the following command to install all modules at once

   ```sh
   pip3 install numpy matplotlib tensorflow
   ```

### Running the code

1. Clone the repository
   ```sh
   git clone https://github.com/martindoff/DC-RNN.git
   ```
2. Go to directory 
   ```sh
   cd DC-RNN-master
   ```
3. Generate data from the dynamical system
   ```python
   python3 generate_data.py
   ``` 
4. Run the program
   ```python
   python3 DC_RNN_model.py
   ```
### Options 
   
1. To load an existing model, set the `load` variable in `DC_RNN_model.py` to `True`
```python
   load = True
   ``` 
   
   Set the variable to `False` if the model has to be (re)trained. 


## Application

Such model have applications, e.g. in the framework of robust tube MPC
for systems representable as a difference of convex functions (see paper on [DC-TMPC](https://ora.ox.ac.uk/objects/uuid:a3a0130b-5387-44b3-97ae-1c9795b91a42/download_file?safe_filename=Doff-Sotta_and_Cannon_2022_Difference_of_convex.pdf&file_format=application%2Fpdf&type_of_work=Conference+item)) 
The DC-RNN model allows one to learn the dynamics in DC form and then apply the DC-TMPC algorithm.

<!-- CONTACT -->
## Contact

Martin Doff-Sotta - martin.doff-sotta@eng.ox.ac.uk

Linkedin: https://www.linkedin.com/in/mdoffsotta/



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
