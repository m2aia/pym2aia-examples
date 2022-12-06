## Run on system

to run the examples, install [M²aia](https://github.com/m2aia/m2aia) and [set system variables](https://github.com/m2aia/pym2aia) to load the required libraries 

follow this instructions.

1. Follow the [installation instructions](https://github.com/m2aia/pym2aia#readme) on pyM²aia's project site.
2. ``` git clone https://github.com/m2aia/pym2aia-examples ```
3. ``` cd pym2aia-examples ```
4. ``` git submodule update --recursive --init ```
5. ``` pip install -r requirements.txt ```

By running an example, the required data set [MTBLS2639](https://www.ebi.ac.uk/metabolights/MTBLS2639/) will be downloaded automatically. 
This will take some time (e.g. in Example I only one slice of four is downloaded, that takes about 12 minutes).

## Run in Docker

Prepare source files
1. ``` git clone https://github.com/m2aia/pym2aia-examples ``` 
2. ``` cd pym2aia-examples ```


Build without gpu support (Example I-III)
1. ``` docker build -t pym2aia-examples -f Dockerfile . ```
2. ``` docker run -ti --rm -v $(pwd):/examples pym2aia-examples Example_I.ipynb $(id -u $USER)```


Build with gpu support (Example IV-VI)
1. ``` docker build -t pym2aia-examples -f Dockerfile.gpu . ```
2. ``` docker run -ti --rm --gpus all -v $(pwd):/examples pym2-gpu Example_IV_A.ipynb $(id -u $USER)```


if the last argument ```$(id -u $USER)``` is set, all items in ```$(pwd)/data```, ```$(pwd)/results``` and ```$(pwd)/models``` will change ownership to the current user (otherwise files will be owned by the root user).



