MODEL_DIR=models
RESULT_DIR=results
DATA_DIR=data 

cd /examples

# arg 1> Filename
# arg 2> UserID

if [ "$2" ]; then
  echo "Start notebook $1"
  jupyter nbconvert \
    --ExecutePreprocessor.allow_errors=True \
    --ExecutePreprocessor.timeout=-1 \
    --FilesWriter.build_directory=$RESULT_DIR \
    --to markdown \
    --execute $1
fi

if [ "$2" ]; then
  echo "Change ownership of all documents to id $2 in..."
  if [ -d "$RESULT_DIR" ]; then 
    chown -R $2:$2 $RESULT_DIR
    echo "...$RESULT_DIR"
  fi

  if [ -d "$DATA_DIR" ]; then 
    chown -R $2:$2 $DATA_DIR
    echo "...$DATA_DIR"
  fi
  
  if [ -d "$MODEL_DIR" ]; then 
    chown -R $2:$2 $MODEL_DIR
    echo "...$MODEL_DIR"
  fi
fi