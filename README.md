# MNIST_GAN
Introduction:
This is a personal project to learn more about the basics of GAN (generative adversarial network) and how they can be applied. The goal is to build a general framework that can work for image datasets of any size, or be compatible with minor changes. I would be training and testing the model using the MNIST dataset, which is pretty basic but will let me easily debug and figure out what went wrong. 

Another important goal of this project is to build well-structured files, with clean, reusable functions as well as well defined files, including config, gitignore, requirements, as well as dedicated folders for inputs. 

# What is GAN?
The basic idea of GAN can be described using a simple example. Imagine someone tries to create counterfiet bills. The police department quickly figures out the difference between the real and counterfiet bills and stops them from being circulated. Soon, a new set of more sophisticated counterfiets were created, and the police department has to once again figure out how to differenciate between real and counterfiet. The end results is, as we repeat the process, we eventually reach a point where the counterfiets are almost indistinguishable from the real ones (assuming he does not get caught before then)

# Project


# Edit 1: (July 5, 2025)
Somewhere along the way this became a Pytorch-lightning learning project, which is fine since its also a useful library to know. 
pytorch-lightning is a pretty useful library for cutting down / cleaning code, grouping relevant code chunks together and removing a lot of boilerplate codes, such as training loops, from the files.

- The core part of lightning is the LightningModule, which captures a lot of the usual training loop, as well as automatic optimizer configuration 