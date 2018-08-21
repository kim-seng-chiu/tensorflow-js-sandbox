function fibonacci(num){
    let a = 1, b = 0, temp;
    let seq = [];
    while(num > 0){
        temp = a;
        a += b;
        b = temp;
        seq.push(b);
        num--;
    }
    return seq;
}

const fibs = fibonacci(100);
console.log(fibs);
const xs = tf.tensor1d(fibs.slice(0,fibs.length - 1));
const ys = tf.tensor1d(fibs.slice(1));

const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin); //subtract - tf function

function norm(x){
    return x.sub(xmin).div(xrange);
}

xsNorm = norm(xs);
ysNorm = norm(ys);

//Initialise variables
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));

function predict(x){
    return tf.tidy(() => {
        return a.mul(x).add(b)
    });
}

function loss(predictions, labels){
    return predictions.sub(labels).square().mean();
}

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

const numIterations = 10000;
const errors = [];
for(let iter = 0; iter < numIterations; iter++){
    optimizer.minimize(() => {
        const predYs = predict(xsNorm);
        const e = loss(predYs, ysNorm);
        errors.push(e.dataSync());
        return e;
    })
}

console.log(errors[0]);
console.log(errors[numIterations - 1]);

xTest = tf.tensor1d([2, 354224848179262000000]);
predict(xTest).print();

a.print();
b.print();