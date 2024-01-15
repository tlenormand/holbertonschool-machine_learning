import * as tf from '@tensorflow/tfjs-node';

async function loadModel() {
    // const weights = await tf.loadLayersModel('file://./models/model.weights.bin.index');
    const model = await tf.loadLayersModel('file://./models/model.json');

    return model;
}

const model = await loadModel();
console.log(model.summary());

// Vous pouvez maintenant utiliser le modèle pour effectuer des prédictions
