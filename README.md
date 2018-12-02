# VAE_ABIDE1
The model stored here corresponds to the study "Identification of Autism Spectrum Disorders on Brain Structural MRI with Variational Autoencoders":
(site IHC: poster + )

The architecture of the model in this git repo is: 


To use the model: 

- Install Niftynet (https://niftynet.readthedocs.io/en/dev/installation.html)

- Clone blackPacha/vae_abide1/* in your niftynet/extensions; my_collection_network is a python module repository.
(more info: https://niftynet.readthedocs.io/en/dev/extending_net.html)

- Train from niftynet/extensions with command line: 

net_autoencoder train -c my_collection_network/vae_config.ini --name my_collection_network.my_vae.VAE

- Encode and Analyze from niftynet/extensions with command line:

./encode_and_analyze.sh model_number target.npy outdirname

with: 
  - model_number: the number of the model to use;
  - target.npy: a numpy array (n,) being the labels of the n encoded images;
  - outdirname: the absolute path of the directory you want your output goes to
  
- This will produce in outdirname:
  - a directory with all the encoded files
  - an X.npy file which corresponds to the array of all the encoded features ( dim(X) = (number of images, number of latent features) )
  - ROC AUC scores between X and the target before/after selection of features 
