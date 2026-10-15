#version 450
// Reactive mask, for game draws whose own fragment shader isn't needed (animated people and animals, water):
// alpha 1, which the pipeline's blend multiplies by the class strength (blend constant alpha).
layout(location = 0) out vec4 outMask;

void main() {
    outMask = vec4(0.0, 0.0, 0.0, 1.0);
}
