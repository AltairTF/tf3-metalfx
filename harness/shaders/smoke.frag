#version 450
// --smoke: a soft, flickering puff in front of the wall, blended and without depth writes, like TF3's
// particles: it has no motion vectors, so it tests the layer's reactive mask (MetalFX).
layout(location = 0) in vec3 worldPos;
layout(push_constant) uniform Puff { vec4 centreRadiusAlpha; } puff;  // xy centre, z radius, w alpha
layout(location = 0) out vec4 outColor;

void main() {
    float d = length(worldPos.xy - puff.centreRadiusAlpha.xy) / puff.centreRadiusAlpha.z;
    float a = puff.centreRadiusAlpha.w * clamp(1.0 - d, 0.0, 1.0);
    if (a <= 0.0) discard;
    outColor = vec4(vec3(1.6), a);
}
