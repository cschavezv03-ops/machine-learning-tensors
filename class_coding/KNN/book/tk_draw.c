#include "tk_draw.h"

// -----------------------------------------------------------------------------
// Funciones para el renderizado de gráficas
// -----------------------------------------------------------------------------

// 1. Dibuja líneas continuas conectando los puntos (Polílinea)
void draw_polyline(Tcl_Interp *interp, NDArrayHandle x, NDArrayHandle y, TkViewContext ctx, const char *color, int width) {
    float *x_ptr = (float*)asarray(x);
    float *y_ptr = (float*)asarray(y);
    int n = len(x);
    char cmd[256];

    for (int i = 0; i < n - 1; i++) {
        snprintf(cmd, sizeof(cmd), ".c create line %.0f %.0f %.0f %.0f -fill %s -width %d",
                 MX_P(x_ptr[i], ctx.ox, ctx.min_x, ctx.scale_x),
                 MY_P(y_ptr[i], ctx.oy, ctx.min_y, ctx.scale_y),
                 MX_P(x_ptr[i+1], ctx.ox, ctx.min_x, ctx.scale_x),
                 MY_P(y_ptr[i+1], ctx.oy, ctx.min_y, ctx.scale_y),
                 color, width);
        EVAL_TCL(interp, cmd);
    }
}

// 2. Dibuja marcadores (puntos/óvalos) sobre un conjunto de coordenadas
void draw_markers(Tcl_Interp *interp, NDArrayHandle x, NDArrayHandle y, TkViewContext ctx, float radius, const char *fill, const char *outline) {
    float *x_ptr = (float*)asarray(x);
    float *y_ptr = (float*)asarray(y);
    int n = len(x);
    char cmd[256];

    for (int i = 0; i < n; i++) {
        float cx = MX_P(x_ptr[i], ctx.ox, ctx.min_x, ctx.scale_x);
        float cy = MY_P(y_ptr[i], ctx.oy, ctx.min_y, ctx.scale_y);
        snprintf(cmd, sizeof(cmd), ".c create oval %.0f %.0f %.0f %.0f -fill %s -outline %s",
                 cx - radius, cy - radius, cx + radius, cy + radius, fill, outline);
        EVAL_TCL(interp, cmd);
    }
}

// 3. Renderiza una grilla bidimensional de líneas a partir de tensores de ticks
void draw_grid_lines(Tcl_Interp *interp, NDArrayHandle gx_ticks, NDArrayHandle gy_ticks, TkViewContext ctx, float max_x, float max_y, const char *color, const char *dash) {
    float *tx = (float*)asarray(gx_ticks);
    float *ty = (float*)asarray(gy_ticks);
    int nx = len(gx_ticks), ny = len(gy_ticks);
    char cmd[256];

    for (int i = 0; i < nx; i++) {
        snprintf(cmd, sizeof(cmd), ".c create line %.0f %.0f %.0f %.0f -fill %s -dash %s",
                 MX_P(tx[i], ctx.ox, ctx.min_x, ctx.scale_x), MY_P(ctx.min_y, ctx.oy, ctx.min_y, ctx.scale_y),
                 MX_P(tx[i], ctx.ox, ctx.min_x, ctx.scale_x), MY_P(max_y, ctx.oy, ctx.min_y, ctx.scale_y),
                 color, dash);
        EVAL_TCL(interp, cmd);
    }

    for (int j = 0; j < ny; j++) {
        snprintf(cmd, sizeof(cmd), ".c create line %.0f %.0f %.0f %.0f -fill %s -dash %s",
                 MX_P(ctx.min_x, ctx.ox, ctx.min_x, ctx.scale_x), MY_P(ty[j], ctx.oy, ctx.min_y, ctx.scale_y),
                 MX_P(max_x, ctx.ox, ctx.min_x, ctx.scale_x), MY_P(ty[j], ctx.oy, ctx.min_y, ctx.scale_y),
                 color, dash);
        EVAL_TCL(interp, cmd);
    }
}

// 4. Renderiza grilla con etiquetas de ejes (ideal para diagramas de $0.0$ a $1.0$)
void draw_grid_with_ticks(Tcl_Interp *interp, NDArrayHandle ticks, TkViewContext ctx) {
    float *t_vals = (float*)asarray(ticks);
    int n = len(ticks);
    char cmd[512];

    for (int i = 0; i < n; i++) {
        float t = t_vals[i];
        snprintf(cmd, sizeof(cmd),
                 ".c create line %.0f %.0f %.0f %.0f -fill #e0e0e0 -dash {2 2}; "
                 ".c create line %.0f %.0f %.0f %.0f -fill #e0e0e0 -dash {2 2}; "
                 ".c create text %.0f %.0f -text \"%.1f\" -font {Helvetica 9}; "
                 ".c create text %.0f %.0f -text \"%.1f\" -font {Helvetica 9} -anchor e",
                 MX_P(0, ctx.ox, 0, ctx.scale_x), MY_P(t, ctx.oy, 0, ctx.scale_y), 
                 MX_P(1, ctx.ox, 0, ctx.scale_x), MY_P(t, ctx.oy, 0, ctx.scale_y),
                 MX_P(t, ctx.ox, 0, ctx.scale_x), MY_P(0, ctx.oy, 0, ctx.scale_y), 
                 MX_P(t, ctx.ox, 0, ctx.scale_x), MY_P(1, ctx.oy, 0, ctx.scale_y),
                 MX_P(t, ctx.ox, 0, ctx.scale_x), MY_P(-0.04f, ctx.oy, 0, ctx.scale_y), t,
                 MX_P(-0.03f, ctx.ox, 0, ctx.scale_x), MY_P(t, ctx.oy, 0, ctx.scale_y), t);
        EVAL_TCL(interp, cmd);
    }
}

// 5. Renderiza mapa de calor (Heatmap/PDF) a partir de coordenadas e intensidades
void draw_heatmap(Tcl_Interp *interp, NDArrayHandle x, NDArrayHandle y, NDArrayHandle Z, TkViewContext ctx, float box_half_size, float intensity_scale) {
    float *x_ptr = (float*)asarray(x);
    float *y_ptr = (float*)asarray(y);
    float *z_ptr = (float*)asarray(Z);
    int n = len(Z);
    char cmd[256];

    for (int i = 0; i < n; i++) {
        int c = (int)(z_ptr[i] * intensity_scale);
        c = (c > 255) ? 255 : ((c < 0) ? 0 : c);

        float cx = MX_P(x_ptr[i], ctx.ox, ctx.min_x, ctx.scale_x);
        float cy = MY_P(y_ptr[i], ctx.oy, ctx.min_y, ctx.scale_y);

        snprintf(cmd, sizeof(cmd), ".c create rect %.0f %.0f %.0f %.0f -fill #%02x%02x%02x -outline {}",
                 cx - box_half_size, cy - box_half_size, 
                 cx + box_half_size, cy + box_half_size, 
                 c, 40, 255 - c);
        EVAL_TCL(interp, cmd);
    }
}
