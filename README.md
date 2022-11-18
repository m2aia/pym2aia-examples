# Run pym2aia-examples

## Docker

Run with build contex as the root of this direcory.

``` git clone https://github.com/m2aia/pym2aia-examples ``` 
``` cd pym2aia-examples ```
``` docker build -t pym2aia-examples . ```
``` docker run -ti --rm -v $(pwd):/examples pym2aia-examples Example_I.ipynb $(id -u $USER)```

if the last argument ```$(id -u $USER)``` is set, all items in ```$(pwd)/data```, ```$(pwd)/results``` and ```$(pwd)/models``` will change ownership to the current user (otherwise files will be owned by the root user).


to run the examples install [M²aia](https://github.com/m2aia/m2aia) and follow this instructions.

1. ``` pip install git+https://github.com/m2aia/pym2aia.git ```
2. ``` git clone https://github.com/m2aia/pym2aia-examples ```
3. ``` cd pym2aia-examples ```
4. ``` git submodule update --recursive --init ```
5. ``` pip install -r requirements.txt ```

By running an example, the required data set [MTBLS2639](https://www.ebi.ac.uk/metabolights/MTBLS2639/) will be downloaded automatically. 
This will take some time (e.g. in Example I only one slice of four is downloaded, that takes about 12 minutes).
