#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <math.h> // NAN e INFINITY
#include <stdlib.h>
#include <regex.h>
#include "tk_draw.h"
#include "sml.h"
#include "Tensors.h"

// 1.1. Helper para Label Encoding de strings
static float str_column_to_int(CSV_Container* res, const char* str, int len) {
    for (int i = 0; i < res->label_count; i++)
        if (strncmp(res->labels[i], str, len) == 0 && res->labels[i][len] == '\0') return (float)i;
    
    // +2: Espacio para el nuevo elemento Y para el sentinela NULL final
    res->labels = realloc(res->labels, (res->label_count + 2) * sizeof(char*));
    res->labels[res->label_count] = strndup(str, len);
    res->labels[res->label_count + 1] = NULL; // Sentinela NULL obligado
    return (float)(res->label_count++);
}

// 1.2. Function for loading a .csv file into a tensor
// Función autorizada para el uso de bucles for por tratarse de la letura de escalares en el archivo .csv.
CSV_Container load_csv(const char* file_path, const char* delimiter, int has_header) {
    CSV_Container res = {0};
    FILE* fh = fopen(file_path, "r");
    if (!fh) return (fprintf(stderr, "Error al abrir %s\n", file_path), res);

    // Coincide con secuencias que NO contienen los delimitadores ni saltos de línea
    char pat[256];
    snprintf(pat, sizeof(pat), "[^%s\r\n]+", delimiter ? delimiter : ",;\t");
    regex_t re;
    regcomp(&re, pat, REG_EXTENDED);

    char line[4096];
    int cap = 256, total = 0;
    float* buf = malloc(cap * sizeof(float));
    regmatch_t m;

    // --- Procesamiento de header con Regex ---
    if (has_header && fgets(line, sizeof(line), fh)) {
        char* p = line;
        int h_cap = 10;
        res.header = malloc((h_cap + 1) * sizeof(char*)); // +1 para el sentinela NULL
        while (regexec(&re, p, 1, &m, 0) == 0) {
            if (res.header_count >= h_cap) {
                h_cap *= 2;
                res.header = realloc(res.header, (h_cap + 1) * sizeof(char*));
            }
            res.header[res.header_count++] = strndup(p + m.rm_so, m.rm_eo - m.rm_so);
            p += m.rm_eo;
        }
        res.header[res.header_count] = NULL; // Sentinela NULL al final del header
    }

    // --- Procesamiento de datos numéricos y texto (categorical encoding) ---
    while (fgets(line, sizeof(line), fh)) {
        char* p = line;
        int cols = 0;
        while (regexec(&re, p, 1, &m, 0) == 0) {
            if (total >= cap) buf = realloc(buf, (cap *= 2) * sizeof(float));
            
            char* token = p + m.rm_so;
            char* endptr;
            float val = strtof(token, &endptr);
            
            // Si strtof no pudo parsear nada (endptr == token), es una cadena de texto
            if (endptr == token) {
                val = str_column_to_int(&res, token, m.rm_eo - m.rm_so);
            }

            buf[total++] = val;
            cols++;
            p += m.rm_eo;
        }
        if (cols > 0) { res.cols = cols; res.rows++; }
    }

    fclose(fh);
    regfree(&re);
    
    if (total > 0) res.dataset = array(buf, shape(res.rows, res.cols));
    free(buf);
    return res;
}

// 1.3. Liberación de memoria del objeto CSV_Container
void free_csv_container(CSV_Container* res) {
    if (res->header) {
        for (int i = 0; i < res->header_count; i++) free(res->header[i]);
        free(res->header);
    }
    if (res->labels) {
        for (int i = 0; i < res->label_count; i++) free(res->labels[i]);
        free(res->labels);
    }
    if (res->dataset) free_tensors(res->dataset);
}

// 2.1. Calcula los valores mínimo y máximo por columna
NDArrayHandle dataset_minmax(NDArrayHandle dataset) {
    NDArrayHandle min_vals = min(dataset, axis(0));
    NDArrayHandle max_vals = max(dataset, axis(0));
    NDArrayHandle stacked  = stack(0, min_vals, max_vals);
    NDArrayHandle result   = transpose(stacked);

    // Libera los tensores temporales de cálculo intermedio
    free_tensors(min_vals, max_vals, stacked);
    return result;
}

// 2.2. Normaliza el dataset al rango 0-1 (In-place)
void normalize_dataset(NDArrayHandle X, NDArrayHandle minmax) {
    NDArrayHandle min_vals = slice(minmax, ":", 0);
    NDArrayHandle max_vals = slice(minmax, ":", 1);
    
    NDArrayHandle diff_X   = sub(X, min_vals);
    NDArrayHandle range    = sub(max_vals, min_vals);
    NDArrayHandle norm     = div(diff_X, range);
    
    copyto(norm, X); // Copia in-place al tensor X

    // Libera todos los tensores temporales creados en el cálculo
    free_tensors(min_vals, max_vals, diff_X, range, norm);
}

// 2.4. Estandariza el dataset (In-place)
void standardize_dataset(NDArrayHandle X, NDArrayHandle means, NDArrayHandle stdevs) {
    NDArrayHandle diff  = sub(X, means);
    NDArrayHandle std_X = div(diff, stdevs);
    
    copyto(std_X, X); // Copia in-place

    // Libera tensores auxiliares
    free_tensors(diff, std_X);
}

// ==============================================================================
// 3.1. Train and test split
// ==============================================================================
/**
 * Divide un dataset en conjuntos de entrenamiento (train) y prueba (test).
 * Retorna un NDArrayList con [train, test].
 */
NDArrayList train_test_split(NDArrayHandle dataset, float split_ratio) {

    int train_size = (int)(split_ratio * len(dataset));

    // 1. Generación y mezcla aleatoria de índices vectorizada
    NDArrayHandle idx = shuffle(arange(len(dataset)));

    // 2. Extracción vectorizada de rangos de índices (train y test)
    NDArrayHandle train_idx = slice(idx, range(0, train_size));
    NDArrayHandle test_idx  = slice(idx, range(train_size, len(dataset)));

    // 3. Extracción vectorizada de filas mediante take
    NDArrayHandle train = take(dataset, train_idx);
    NDArrayHandle test  = take(dataset, test_idx);

    // 4. Liberación de tensores temporales de índices
    free_tensors(idx, train_idx, test_idx);

    // Retorna el contenedor [train, test]
    return tolist(train, test);
}

// ==============================================================================
// 3.2. k-fold cross-validation split
// ==============================================================================
/**
 * Divide un dataset en K folds creando directamente un tensor 3D: [n_folds, fold_size, cols].
 * Reemplaza todo bucle for + push + stack por un único 'take' y 'reshape'.
 */
NDArrayHandle cross_validation_split(NDArrayHandle dataset, int n_folds) {

    int fold_size = len(dataset) / n_folds;
    int total     = n_folds * fold_size;

    // 1. Generar índices aleatorios y truncar al total divisible exactamente por n_folds
    NDArrayHandle idx_full   = arange(len(dataset));
    NDArrayHandle idx_rand   = shuffle(idx_full);
    NDArrayHandle idx_sliced = slice(idx_rand, range(0, total));

    // 2. Extracción vectorizada de filas
    NDArrayHandle taken = take(dataset, idx_sliced);

    // 3. Redimensionar a tensor 3D [n_folds, fold_size, cols] con inferencia automática (-1)
    NDArrayHandle folds = reshape(taken, shape(n_folds, fold_size, -1));

    // 4. Limpieza de tensores intermedios
    free_tensors(idx_full, idx_rand, idx_sliced, taken);

    return folds;
}

// ==============================================================================
// 3.3. Conteo de etiquetas vectorizado (Soporta datasets 2D y folds 3D)
// ==============================================================================
/**
 * Calcula la frecuencia por clase mediante One-Hot Encoding y reducciones matriciales.
 * Determina automáticamente el eje de reducción según la dimensionalidad del dataset.
 * 
 * @param dataset Matrix 2D [rows, cols] o Tensor 3D de folds [n_folds, fold_size, cols].
 * @param num_classes Cantidad de clases. Si es <= 0, se infiere automáticamente.
 */
NDArrayHandle count_labels(NDArrayHandle dataset, int num_classes) {
    // 1. Extrae la última columna (etiquetas) y garantiza tipo entero int32
    NDArrayHandle Y_raw = slice(dataset, "...", -1);
    NDArrayHandle Y     = astype(Y_raw, "int32");

    // 2. Inferir num_classes si no fue especificado
    if (num_classes <= 0) {
        NDArrayHandle max_t = max(Y);
        num_classes = asscalar(max_t) + 1;
        free_tensors(max_t);
    }

    // 3. One-Hot Encoding vectorizado: convierte vector de clases en matriz binaria
    NDArrayHandle one_hot_matrix = one_hot(Y, num_classes);

    // 4. Inferencia del eje mediante ndim(dataset):
    //    Axis 1 para tensores 3D (folds) | Axis 0 para matrices 2D (datasets simples)
    NDArrayHandle counts = sum(one_hot_matrix, axis(ndim(dataset) == 3 ? 1 : 0));

    // 5. Liberar tensores temporales
    free_tensors(Y_raw, Y, one_hot_matrix);

    return counts;
}

// ==============================================================================
// 3.4. Helper de impresión formateada.
// Función autorizada para el uso de bucles for por tratarse de impresión de los escalares.
// ==============================================================================
void print_formatted_counts(NDArrayHandle counts, char** labels, int label_count) {
    float* data = asarray(counts);
    int total_rows = len(counts);

    // Caso 1D: Un solo conteo global (Dataset completo, train o test)
    if (total_rows == label_count) {
        for (int i = 0; i < label_count; i++) {
            const char* name = (labels && labels[i]) ? labels[i] : "Clase";
            printf("%s => %.0f  ", name, data[i]);
        }
        printf("\n");
    } else {
        // Caso 2D: Múltiples folds [n_folds, label_count]
        for (int f = 0; f < total_rows; f++) {
            printf("Fold %2d: ", f + 1);
            for (int c = 0; c < label_count; c++) {
                const char* name = (labels && labels[c]) ? labels[c] : "Clase";
                printf("%s => %.0f  ", name, data[f * label_count + c]);
            }
            printf("\n");
        }
    }
}

// =============================================================================
// 4.1. Accuracy Metric
// =============================================================================
float accuracy_metric(NDArrayHandle actual, NDArrayHandle predicted) {
    NDArrayHandle cmp = eq(predicted, actual);
    NDArrayHandle cmp_int = astype(cmp, "int64");
    NDArrayHandle cmp_sum = sum(cmp_int);

    float correct = asscalar(cmp_sum);
    int total_len = len(actual);

    free_tensors(cmp, cmp_int, cmp_sum);

    return (100.0 * correct) / total_len;
}

// =============================================================================
// 4.2. Confusion Matrix
// =============================================================================
// Retorna un NDArrayList con [clases, matrix]
NDArrayList confusion_matrix(NDArrayHandle actual, NDArrayHandle predicted) {
    // 1. Obtención del valor máximo y cálculo del número de clases (max + 1)
    NDArrayHandle max_tensor = max(actual);
    int num_classes = (int)asscalar(max_tensor) + 1;

    // 2. Generación vectorial de la secuencia de clases [0, 1, ..., num_classes - 1]
    NDArrayHandle classes = arange(num_classes);

    // 3. Codificación One-Hot de los tensores de entrada
    NDArrayHandle actual_oh    = one_hot(actual, num_classes);
    NDArrayHandle predicted_oh = one_hot(predicted, num_classes);

    // 4. Producto matricial: (actual_oh^T) * predicted_oh
    NDArrayHandle actual_oh_T = transpose(actual_oh);
    NDArrayHandle matrix      = dot(actual_oh_T, predicted_oh);

    // 5. Empaquetado en NDArrayList
    NDArrayList result = declare(2);
    idx(result, 0) = classes;
    idx(result, 1) = matrix;

    // Liberación de tensores intermedios
    free_tensors(max_tensor, actual_oh, predicted_oh, actual_oh_T);

    return result;
}

// =============================================================================
// 4.3. Mean Absolute Error (MAE)
// =============================================================================
float mae_metric(NDArrayHandle actual, NDArrayHandle predicted) {
    NDArrayHandle diff = sub(actual, predicted);
    NDArrayHandle abs_diff = abs(diff);
    NDArrayHandle sum_val = sum(abs_diff);

    float mae = asscalar(sum_val) / len(actual);

    free_tensors(diff, abs_diff, sum_val);

    return mae;
}

// =============================================================================
// 4.4. Root Mean Squared Error (RMSE)
// =============================================================================
float rmse_metric(NDArrayHandle actual, NDArrayHandle predicted) {
    NDArrayHandle diff = sub(actual, predicted);
    NDArrayHandle squared = square(diff);
    NDArrayHandle sum_val = sum(squared);

    float mean_error = asscalar(sum_val) / len(actual);
    float rmse = sqrtf(mean_error);

    free_tensors(diff, squared, sum_val);

    return rmse;
}

// =============================================================================
// 4.5. Performance Metrics (ROC Curves)
// =============================================================================
// Retorna un NDArrayList con [fprs, tprs]
NDArrayList perf_metrics(NDArrayHandle actual, NDArrayHandle predicted_prob, NDArrayHandle threshold, float pos_cls) {
    // Expandir dimensiones para Broadcasting: y_prob (N, 1), thresh (1, K)
    NDArrayHandle y_prob = expand_dims(predicted_prob, 1);
    NDArrayHandle thresh = expand_dims(threshold, 0);

    // Matriz binaria de predicciones (N, K)
    NDArrayHandle pred_matrix = (pos_cls == 1.0) ? ge(y_prob, thresh) : le(y_prob, thresh);
    NDArrayHandle pred_matrix_f32 = astype(pred_matrix, "float32");

    // Máscaras de etiquetas reales (N, 1)
    NDArrayHandle eq_pos = eq(actual, pos_cls);
    NDArrayHandle ne_pos = ne(actual, pos_cls);
    
    NDArrayHandle is_pos = expand_dims(astype(eq_pos, "float32"), 1);
    NDArrayHandle is_neg = expand_dims(astype(ne_pos, "float32"), 1);

    // Conteo de TP y FP por columna/umbral (K,)
    NDArrayHandle pos_pred = mul(is_pos, pred_matrix_f32);
    NDArrayHandle neg_pred = mul(is_neg, pred_matrix_f32);

    NDArrayHandle tp = sum(pos_pred, axis(0));
    NDArrayHandle fp = sum(neg_pred, axis(0));

    // Totales absolutos
    NDArrayHandle pos_count_tensor = sum(is_pos);
    NDArrayHandle neg_count_tensor = sum(is_neg);

    float pos_count = asscalar(pos_count_tensor);
    float neg_count = asscalar(neg_count_tensor);

    // Vectores TPR y FPR (K,)
    NDArrayHandle tprs = (pos_count > 0) ? div(tp, pos_count) : zeros_like(threshold);
    NDArrayHandle fprs = (neg_count > 0) ? div(fp, neg_count) : zeros_like(threshold);

    NDArrayList result = declare(2);
    idx(result, 0) = fprs;
    idx(result, 1) = tprs;

    // Liberar tensores intermedios
    free_tensors(y_prob, thresh, pred_matrix, pred_matrix_f32, 
                 eq_pos, ne_pos, is_pos, is_neg, 
                 pos_pred, neg_pred, tp, fp, 
                 pos_count_tensor, neg_count_tensor);

    return result;
}

// =============================================================================
// 4.6. Trapezoidal Rule (Trapz - AUC)
// =============================================================================
float trapz(NDArrayHandle x, NDArrayHandle y) {
    // Slicing: x[1:] y x[0:-1]
    NDArrayHandle x_right = slice(x, range(1, None));
    NDArrayHandle x_left  = slice(x, range(0, -1));
    NDArrayHandle dx      = sub(x_right, x_left);

    NDArrayHandle y_right = slice(y, range(1, None));
    NDArrayHandle y_left  = slice(y, range(0, -1));
    NDArrayHandle y_add   = add(y_right, y_left);
    NDArrayHandle avg_y   = div(y_add, 2.0f);

    NDArrayHandle area_elements = mul(dx, avg_y);
    NDArrayHandle total_sum     = sum(area_elements);

    float auc = asscalar(total_sum);

    free_tensors(x_right, x_left, dx, y_right, y_left, y_add, avg_y, area_elements, total_sum);

    return auc;
}

// =============================================================================
// 4.7. Plot ROC curve
// =============================================================================
void plot_roc_curve(NDArrayHandle sorted_fprs, NDArrayHandle sorted_tprs, float auc) {
    Tcl_Interp *interp = Tcl_CreateInterp();
    Tcl_Init(interp); Tk_Init(interp);

    EVAL_TCL(interp, "wm title . \"ROC Curve Plot\"; canvas .c -width 650 -height 650 -bg white; pack .c");
    TkViewContext ctx = { .ox = 100.0f, .oy = 520.0f, .min_x = 0.0f, .min_y = 0.0f, .scale_x = 400.0f, .scale_y = 400.0f };
    char cmd[512];

    // 1. Grid y ticks
    NDArrayHandle ticks = linspace(0.0f, 1.0f, 6);
    draw_grid_with_ticks(interp, ticks, ctx);
    free_tensors(ticks);

    // 2. Configuración de los ejes (X, Y), diagonal y títulos estáticos
    snprintf(cmd, sizeof(cmd),
             ".c create line %.0f %.0f %.0f %.0f -fill black -width 2; "
             ".c create line %.0f %.0f %.0f %.0f -fill black -width 2; "
             ".c create line %.0f %.0f %.0f %.0f -fill red -dash {4 4} -width 2; "
             ".c create text %.0f %.0f -text \"ROC Curve (AUC = %.2f)\" -font {Helvetica 14 bold}; "
             ".c create text %.0f %.0f -text \"Specificity: False Positive Rate (FPR)\" -font {Helvetica 11 bold}; "
             ".c create text %.0f %.0f -text \"Sensitivity: True Positive Rate (TPR)\" -font {Helvetica 11 bold} -angle 90",
             MX_P(0, ctx.ox, 0, ctx.scale_x), MY_P(0, ctx.oy, 0, ctx.scale_y), MX_P(1, ctx.ox, 0, ctx.scale_x), MY_P(0, ctx.oy, 0, ctx.scale_y),
             MX_P(0, ctx.ox, 0, ctx.scale_x), MY_P(0, ctx.oy, 0, ctx.scale_y), MX_P(0, ctx.ox, 0, ctx.scale_x), MY_P(1, ctx.oy, 0, ctx.scale_y),
             MX_P(0, ctx.ox, 0, ctx.scale_x), MY_P(0, ctx.oy, 0, ctx.scale_y), MX_P(1, ctx.ox, 0, ctx.scale_x), MY_P(1, ctx.oy, 0, ctx.scale_y),
             MX_P(0.5f, ctx.ox, 0, ctx.scale_x), MY_P(1.12f, ctx.oy, 0, ctx.scale_y), auc,
             MX_P(0.5f, ctx.ox, 0, ctx.scale_x), MY_P(-0.12f, ctx.oy, 0, ctx.scale_y),
             MX_P(-0.14f, ctx.ox, 0, ctx.scale_x), MY_P(0.5f, ctx.oy, 0, ctx.scale_y));
    EVAL_TCL(interp, cmd);

    // 3. Curva ROC: Renderizado de líneas y marcadores
    draw_polyline(interp, sorted_fprs, sorted_tprs, ctx, "blue", 2);
    draw_markers(interp, sorted_fprs, sorted_tprs, ctx, 3.0f, "blue", "white");

    printf("=== Mostrando ventana Tk de la curva ROC ===\n");
    Tk_MainLoop();
    Tcl_DeleteInterp(interp);
}