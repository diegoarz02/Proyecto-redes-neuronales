"""
Training pipeline: entrena Dense, CNN, RNN en Fashion-MNIST.
Genera gráficas de comparación y comportamiento de red.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

from models import build_dense_model, build_cnn_model, build_rnn_model, CLASS_NAMES

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
EPOCHS = 5
BATCH_SIZE = 128


def load_data():
    """Carga Fashion-MNIST y prepara datos para cada red."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    data = {
        'dense': {
            'x_train': x_train.reshape(-1, 784),
            'x_test': x_test.reshape(-1, 784),
        },
        'cnn': {
            'x_train': x_train.reshape(-1, 28, 28, 1),
            'x_test': x_test.reshape(-1, 28, 28, 1),
        },
        'rnn': {
            'x_train': x_train,  # (samples, 28, 28) - ya en forma de secuencia
            'x_test': x_test,
        },
        'y_train': y_train,
        'y_test': y_test,
        'x_train_raw': x_train,
        'x_test_raw': x_test,
    }
    return data


def train_all_models(data):
    """Entrena los 3 modelos y devuelve historiales."""
    models_info = {}

    # --- Dense/MLP ---
    print("\n" + "="*60)
    print("  Entrenando Dense/MLP...")
    print("="*60)
    dense_model = build_dense_model()
    dense_history = dense_model.fit(
        data['dense']['x_train'], data['y_train'],
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, verbose=1
    )
    models_info['Dense_MLP'] = {
        'model': dense_model,
        'history': dense_history.history,
        'x_test': data['dense']['x_test'],
    }

    # --- CNN ---
    print("\n" + "="*60)
    print("  Entrenando CNN...")
    print("="*60)
    cnn_model = build_cnn_model()
    cnn_history = cnn_model.fit(
        data['cnn']['x_train'], data['y_train'],
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, verbose=1
    )
    models_info['CNN'] = {
        'model': cnn_model,
        'history': cnn_history.history,
        'x_test': data['cnn']['x_test'],
    }

    # --- RNN/LSTM ---
    print("\n" + "="*60)
    print("  Entrenando RNN (LSTM)...")
    print("="*60)
    rnn_model = build_rnn_model()
    rnn_history = rnn_model.fit(
        data['rnn']['x_train'], data['y_train'],
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=0.2, verbose=1
    )
    models_info['RNN_LSTM'] = {
        'model': rnn_model,
        'history': rnn_history.history,
        'x_test': data['rnn']['x_test'],
    }

    return models_info


# ==========================================
#   GRÁFICAS DE COMPARACIÓN
# ==========================================

def plot_accuracy_comparison(models_info):
    """Curvas de accuracy entrenamiento/validación superpuestas."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    colors = {'Dense_MLP': '#00d4ff', 'CNN': '#ff6b9d', 'RNN_LSTM': '#c084fc'}

    for name, info in models_info.items():
        h = info['history']
        axes[0].plot(h['accuracy'], label=f'{name}', color=colors[name], linewidth=2)
        axes[0].plot(h['val_accuracy'], label=f'{name} (val)', color=colors[name], linewidth=2, linestyle='--')

        axes[1].plot(h['loss'], label=f'{name}', color=colors[name], linewidth=2)
        axes[1].plot(h['val_loss'], label=f'{name} (val)', color=colors[name], linewidth=2, linestyle='--')

    for ax, title, ylabel in [(axes[0], 'Accuracy por Época', 'Accuracy'),
                               (axes[1], 'Loss por Época', 'Loss')]:
        ax.set_facecolor('#1a1a2e')
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Época', color='#888')
        ax.set_ylabel(ylabel, color='#888')
        ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.2, color='#444')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_loss_comparison.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ accuracy_loss_comparison.png")


def plot_metrics_bars(models_info, y_test):
    """Barras comparativas de accuracy, precision, recall, f1."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')

    metrics_data = {}
    for name, info in models_info.items():
        y_pred = np.argmax(info['model'].predict(info['x_test'], verbose=0), axis=1)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        acc = np.mean(y_pred == y_test)
        metrics_data[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ['#00d4ff', '#ff6b9d', '#c084fc']

    for i, (name, metrics) in enumerate(metrics_data.items()):
        values = [metrics[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, color='white', fontsize=12)
    ax.set_ylabel('Score', color='#888')
    ax.set_title('Comparación de Métricas por Modelo', color='white', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors='#888')
    ax.grid(True, axis='y', alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'metrics_comparison.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ metrics_comparison.png")
    return metrics_data


def plot_confusion_matrices(models_info, y_test):
    """Matriz de confusión por cada modelo."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0f0f1a')

    for idx, (name, info) in enumerate(models_info.items()):
        y_pred = np.argmax(info['model'].predict(info['x_test'], verbose=0), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='magma', vmin=0, vmax=1)
        ax.set_title(name, color='white', fontsize=13, fontweight='bold')

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                color = 'white' if cm_norm[i, j] < 0.5 else 'black'
                ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                        color=color, fontsize=7)

        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=7, color='#aaa')
        ax.set_yticklabels(CLASS_NAMES, fontsize=7, color='#aaa')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ confusion_matrices.png")


# ==========================================
#   GRÁFICAS DE COMPORTAMIENTO DE RED
# ==========================================

def plot_architecture_diagrams(models_info):
    """Diagrama visual de la arquitectura de cada red."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor('#0f0f1a')
    colors_map = {'Dense_MLP': '#00d4ff', 'CNN': '#ff6b9d', 'RNN_LSTM': '#c084fc'}

    for idx, (name, info) in enumerate(models_info.items()):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')
        model = info['model']
        layer_names = []
        layer_params = []

        for layer in model.layers:
            lname = layer.name
            config = layer.get_config()
            if hasattr(layer, 'units'):
                desc = f"{lname}\n({layer.units} units)"
            elif hasattr(layer, 'filters'):
                desc = f"{lname}\n({layer.filters} filters)"
            elif 'rate' in config:
                desc = f"Dropout\n(rate={config['rate']})"
            else:
                desc = lname
            layer_names.append(desc)
            layer_params.append(layer.count_params())

        n = len(layer_names)
        y_positions = np.linspace(0.9, 0.1, n)
        max_params = max(layer_params) if max(layer_params) > 0 else 1

        for i, (y, lname, params) in enumerate(zip(y_positions, layer_names, layer_params)):
            width = max(0.15, 0.6 * (params / max_params))
            rect = plt.Rectangle((0.5 - width/2, y - 0.03), width, 0.06,
                                  facecolor=colors_map[name], alpha=0.3 + 0.5 * (params / max_params),
                                  edgecolor=colors_map[name], linewidth=1.5, transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(0.5, y, lname, ha='center', va='center', color='white',
                    fontsize=8, fontweight='bold', transform=ax.transAxes)
            if params > 0:
                ax.text(0.92, y, f'{params:,}', ha='right', va='center',
                        color='#888', fontsize=7, transform=ax.transAxes)

            if i < n - 1:
                ax.annotate('', xy=(0.5, y_positions[i+1] + 0.035),
                           xytext=(0.5, y - 0.035),
                           arrowprops=dict(arrowstyle='->', color='#555', lw=1.5),
                           xycoords='axes fraction', textcoords='axes fraction')

        total_params = sum(layer_params)
        ax.set_title(f'{name}\n({total_params:,} params)', color='white', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'architecture_diagrams.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ architecture_diagrams.png")


def plot_weight_distributions(models_info):
    """Histogramas de distribución de pesos por capa trainable."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0f0f1a')
    colors_map = {'Dense_MLP': '#00d4ff', 'CNN': '#ff6b9d', 'RNN_LSTM': '#c084fc'}

    for idx, (name, info) in enumerate(models_info.items()):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')
        model = info['model']

        trainable_layers = [l for l in model.layers if len(l.get_weights()) > 0]
        alphas = np.linspace(0.3, 1.0, len(trainable_layers))

        for i, layer in enumerate(trainable_layers):
            weights = layer.get_weights()[0].flatten()
            ax.hist(weights, bins=50, alpha=alphas[i], label=layer.name,
                    color=colors_map[name], edgecolor='none', density=True)

        ax.set_title(f'{name} - Distribución de Pesos', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Valor del Peso', color='#888')
        ax.set_ylabel('Densidad', color='#888')
        ax.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.15, color='#444')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'weight_distributions.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ weight_distributions.png")


def plot_activations_heatmap(models_info, x_sample):
    """Heatmap de activaciones intermedias para una muestra de ejemplo."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0f0f1a')

    samples = {
        'Dense_MLP': x_sample.reshape(1, 784),
        'CNN': x_sample.reshape(1, 28, 28, 1),
        'RNN_LSTM': x_sample.reshape(1, 28, 28),
    }

    for idx, (name, info) in enumerate(models_info.items()):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')
        model = info['model']

        # Get intermediate layer outputs
        dense_layers = [l for l in model.layers if isinstance(l, (keras.layers.Dense,)) and l != model.layers[-1]]
        if len(dense_layers) == 0:
            dense_layers = [l for l in model.layers if hasattr(l, 'units') and l != model.layers[-1]]

        if dense_layers:
            intermediate = keras.Model(inputs=model.inputs, outputs=dense_layers[-1].output)
            activations = intermediate.predict(samples[name], verbose=0)

            if len(activations.shape) == 2:
                act_2d = activations.reshape(-1)
                size = int(np.ceil(np.sqrt(len(act_2d))))
                padded = np.zeros(size * size)
                padded[:len(act_2d)] = act_2d
                act_grid = padded.reshape(size, size)
            else:
                act_grid = activations[0] if len(activations.shape) > 2 else activations

            im = ax.imshow(act_grid if len(act_grid.shape) == 2 else act_grid[:, :act_grid.shape[1]],
                          cmap='inferno', aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(f'{name} - Activaciones', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'activations_heatmap.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ activations_heatmap.png")


def plot_cnn_feature_maps(cnn_model, x_sample):
    """Visualización de feature maps de las capas convolucionales."""
    conv_layers = [l for l in cnn_model.layers if isinstance(l, keras.layers.Conv2D)]

    fig, axes = plt.subplots(len(conv_layers), 8, figsize=(16, 3 * len(conv_layers)))
    fig.patch.set_facecolor('#0f0f1a')
    if len(conv_layers) == 1:
        axes = axes.reshape(1, -1)

    sample = x_sample.reshape(1, 28, 28, 1)

    for row, conv_layer in enumerate(conv_layers):
        intermediate = keras.Model(inputs=cnn_model.inputs, outputs=conv_layer.output)
        feature_maps = intermediate.predict(sample, verbose=0)[0]

        for col in range(min(8, feature_maps.shape[-1])):
            ax = axes[row][col]
            ax.set_facecolor('#1a1a2e')
            ax.imshow(feature_maps[:, :, col], cmap='magma')
            ax.axis('off')
            if col == 0:
                ax.set_title(conv_layer.name, color='white', fontsize=9, fontweight='bold', loc='left')

    plt.suptitle('CNN - Feature Maps (Filtros Convolucionales)', color='white', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cnn_feature_maps.png'), dpi=150, facecolor='#0f0f1a',
                bbox_inches='tight')
    plt.close()
    print("  ✓ cnn_feature_maps.png")


def plot_gradient_magnitudes(models_info, data):
    """Magnitud promedio de gradientes por capa."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#0f0f1a')
    colors_map = {'Dense_MLP': '#00d4ff', 'CNN': '#ff6b9d', 'RNN_LSTM': '#c084fc'}

    sample_data = {
        'Dense_MLP': data['dense']['x_train'][:32],
        'CNN': data['cnn']['x_train'][:32],
        'RNN_LSTM': data['rnn']['x_train'][:32],
    }
    y_sample = data['y_train'][:32]

    for idx, (name, info) in enumerate(models_info.items()):
        ax = axes[idx]
        ax.set_facecolor('#1a1a2e')
        model = info['model']

        x_batch = tf.constant(sample_data[name])
        y_batch = tf.constant(y_sample)

        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=False)
            loss = keras.losses.sparse_categorical_crossentropy(y_batch, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        layer_names_g = []
        grad_mags = []

        for var, grad in zip(model.trainable_variables, gradients):
            if grad is not None:
                short_name = var.name.split('/')[-1].replace(':0', '')
                layer_names_g.append(short_name[:15])
                grad_mags.append(float(tf.reduce_mean(tf.abs(grad)).numpy()))

        bars = ax.barh(range(len(grad_mags)), grad_mags, color=colors_map[name], alpha=0.8)
        ax.set_yticks(range(len(layer_names_g)))
        ax.set_yticklabels(layer_names_g, color='#ccc', fontsize=8)
        ax.set_xlabel('Magnitud Promedio', color='#888')
        ax.set_title(f'{name} - Gradientes', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.grid(True, axis='x', alpha=0.15, color='#444')
        for spine in ax.spines.values():
            spine.set_color('#333')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gradient_magnitudes.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ gradient_magnitudes.png")


def plot_sample_predictions(models_info, x_test_raw, y_test):
    """Muestra predicciones de cada modelo en muestras de ejemplo."""
    fig, axes = plt.subplots(3, 8, figsize=(20, 9))
    fig.patch.set_facecolor('#0f0f1a')

    indices = np.random.choice(len(x_test_raw), 8, replace=False)

    for row, (name, info) in enumerate(models_info.items()):
        y_pred = np.argmax(info['model'].predict(info['x_test'], verbose=0), axis=1)

        for col, i in enumerate(indices):
            ax = axes[row][col]
            ax.set_facecolor('#1a1a2e')
            ax.imshow(x_test_raw[i], cmap='gray')
            ax.axis('off')

            pred_class = CLASS_NAMES[y_pred[i]]
            true_class = CLASS_NAMES[y_test[i]]
            correct = y_pred[i] == y_test[i]
            color = '#4ade80' if correct else '#f87171'

            ax.set_title(f'{pred_class}', color=color, fontsize=8, fontweight='bold')
            if col == 0:
                ax.text(-0.15, 0.5, name, transform=ax.transAxes, color='white',
                       fontsize=10, fontweight='bold', va='center', rotation=90)

    plt.suptitle('Predicciones por Modelo (Verde=Correcto, Rojo=Incorrecto)',
                 color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'sample_predictions.png'), dpi=150, facecolor='#0f0f1a')
    plt.close()
    print("  ✓ sample_predictions.png")


def save_models_and_metrics(models_info, y_test, metrics_data):
    """Guarda modelos y métricas."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = {}
    for name, info in models_info.items():
        model_path = os.path.join(MODELS_DIR, f'{name}.keras')
        info['model'].save(model_path)
        print(f"  ✓ Modelo guardado: {model_path}")

        results[name] = {
            'metrics': metrics_data[name],
            'history': {k: [float(v) for v in vals] for k, vals in info['history'].items()},
        }

    # Determine best model
    best = max(results, key=lambda n: results[n]['metrics']['Accuracy'])
    results['best_model'] = best
    results['class_names'] = CLASS_NAMES

    with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ★ Mejor modelo: {best} (Accuracy: {results[best]['metrics']['Accuracy']:.4f})")


def run_training():
    """Pipeline completo de entrenamiento."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n" + "="*50)
    print("   NEURAL NETWORK COMPARISON PIPELINE")
    print("   Dataset: Fashion-MNIST (70,000 imgs)")
    print("="*50 + "\n")

    # 1. Cargar datos
    print("[1/5] Cargando Fashion-MNIST...")
    data = load_data()
    print(f"  Train: {data['y_train'].shape[0]} muestras | Test: {data['y_test'].shape[0]} muestras")

    # 2. Entrenar modelos
    print("\n[2/5] Entrenando modelos...")
    models_info = train_all_models(data)

    # 3. Gráficas de comparación
    print("\n[3/5] Generando gráficas de comparación...")
    plot_accuracy_comparison(models_info)
    metrics_data = plot_metrics_bars(models_info, data['y_test'])
    plot_confusion_matrices(models_info, data['y_test'])

    # 4. Guardar modelos (Moved up to prevent losing training on plot error)
    print("\n[4/5] Guardando modelos y métricas...")
    save_models_and_metrics(models_info, data['y_test'], metrics_data)

    # 5. Gráficas de comportamiento
    print("\n[5/5] Generando gráficas de comportamiento...")
    try:
        x_sample = data['x_test_raw'][0]
        plot_architecture_diagrams(models_info)
        plot_weight_distributions(models_info)
        plot_activations_heatmap(models_info, x_sample)
        plot_cnn_feature_maps(models_info['CNN']['model'], x_sample)
        plot_gradient_magnitudes(models_info, data)
        plot_sample_predictions(models_info, data['x_test_raw'], data['y_test'])
    except Exception as e:
        print(f"Error generando gráficas de comportamiento: {e}")

    print("\n" + "="*50)
    print("   ✓ PIPELINE COMPLETADO CON ÉXITO")
    print("="*50 + "\n")


if __name__ == '__main__':
    run_training()
