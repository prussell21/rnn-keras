# RNN for Finance - Keras Implementation

Keras/TensorFlow impementation of Recurrent Neural Network for estimating volatility of financial products.



### Example - Yahoo stocks
```
###Calculating True Range day over day.
path = 'GOOG.csv'
df = pd.read_csv(path)
df['rangepct'] = ((df.High - df.Low))/(df.Close.shift(1))
df = df['rangepct'].iloc[1:]
```

### Predicting many to one

Input multiple past daily price ranges to predict next days volatility

#### Example

Input [p1 ,p2, p3 ,p4, p5] --> Output [p6]

Input [p2, p3, p4, p5, p6] --> Output [p7]

Input [p3, p4 ,p5, p6, p7] --> Output [p8]


### Data Preprocessing

Set of functions for splitting, scaling, and reformatting into numpy arrays to feed to RNN model

```
lbr = 5 # look back range
train_pct_split = 0.85

def data_split(df):
    df = df.values.reshape(-1,1)
    train_size = int(len(df) * train_pct_split)
    test_size = len(df) - train_size
    train, test = df[0:train_size,:], df[train_size:len(df),:]
    return train, test

def scale_data(train, test):
    scalar = MinMaxScaler()
    train, test = scalar.fit_transform(train), scalar.fit_transform(test)
    return train, test

def create_dataset(dataset, lbr):
    X, Y = [], []
    for i in range(len(dataset)-lbr-1):
        append_set = dataset[i:(i+lbr), 0]
        X.append(append_set)
        Y.append(dataset[i + (lbr-(lbr-1)), 0])
    return np.array(X), np.array(Y)

def format_input_data(train, test):
    train = train[:,:,newaxis]
    test = test[:,:,newaxis]
    return train, test
```

### Model

For this time series data, a sequential model with multiple LSTM and dropout layers was chosen.

### Prediction and Loss Calculation

The selected evaluation metric for this model is Root Mean Squared Error. Further validation is performed by comparing model scores against a benchmark (output = mean of inputs).


