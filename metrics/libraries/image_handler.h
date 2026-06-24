#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include "../../neural_network/libraries/tensor.h"
#include "../../computer_vision/libraries/scene_graph.h"
#include <stdint.h>

//raw uncompressed RGB stream frame bytes straight into a standard PPM format image.

void Diagnostic_dump_raw_rgb(const char* prefix, unsigned char* raw_rgb_frame, int width, int height);

// Saves normalized tensor image metrics back into an uncompressed gray PGM image,
// writes out flat raw depth floating arrays to an indexable matrix file.
void Diagnostic_dump_depth(const char* prefix, Tensor* preprocessed_tensor, Tensor* depth_prediction);

//Serializes dynamic spatial tracked objects from SceneGraph into a flat structured CSV layout.
void Diagnostic_dump_tracking(const char* prefix, TrackedObject* objects, uint32_t num_objects);

#endif
