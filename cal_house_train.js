function testPattern() {
    console.log("This is working.");
}
function runModel() {
    function dataIngest(){
        // Ingest data here
    }
    
    /* ASSUMPTIONS
        As six variables, polynomial function of degree 6: y = ax^6 + bx^5 + cx^4 + dx^3 + ex^2 + fx + g */
    
    // Set up variables
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));
    const d = tf.variable(tf.scalar(Math.random()));
    const e = tf.variable(tf.scalar(Math.random()));
    const f = tf.variable(tf.scalar(Math.random()));
    const g = tf.variable(tf.scalar(Math.random()));
    
    // Build the model
    function predict(x) {
        return tf.tidy(() => {
            return a.mul(x.pow(tf.scalar(6)))
                .add(b.mul(x.pow(tf.scalar(5))))
                .add(c.mul(x.pow(tf.scalar(4))))
                .add(d.mul(x.pow(tf.scalar(3))))
                .add(e.mul(x.pow(tf.scalar(2))))
                .add(f.mul(x))
                .add(g);
        });
    }
    
    // Define the loss function
    function loss(predictions, labels) {
        const mSE = predictions.sub(labels).square().mean();
        return mSE;
    }
    
    // Optimisation function variables
    const learningRate = 0.5;
    const optim = tf.train.sgd(learningRate);
    
    // Define training loop
    function train(xs, ys, numIterations = 100) {
        for (let iter = 0; iter < numIterations; iter++) {
            optim.minimize(() => {
                const predYs = predict(xs);
                return loss(predYs, ys);
            })
        }
    }
}
