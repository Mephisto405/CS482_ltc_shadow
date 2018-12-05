// bind shadow      {label:"Shadow On", default:true}
// bind shadow_debug {label:"Debug mode", default:false}
// bind second_obstacle {label: "Second Obstacle", default:false}

// bind targetu     {label:"Target u", default:0.0, min:-1.0, max:1.0, step:0.01}
// bind targetv     {label:"Target v", default:0.0, min:-1.0, max:1.0, step:0.01}

// bind width_obstacle       {label:"Obstacle Width",  default: 8, min:0.1, max:15, step:0.1}
// bind height_obstacle      {label:"Obstacle Height", default: 8, min:0.1, max:15, step:0.1}
// bind roty_obstacle        {label:"Obstacle Rotation Y", default: 0, min:0, max:1, step:0.001}
// bind rotz_obstacle        {label:"Obstacle Rotation Z", default: 0, min:0, max:1, step:0.001}
// bind posx                 {label:"Obstacle Position X", default: 5, min:0, max:10, step:0.1}

// bind roughness   {label:"Roughness", default:0.25, min:0.01, max:1, step:0.001}
// bind dcolor      {label:"Diffuse Color",  r:1.0, g:1.0, b:1.0}
// bind scolor      {label:"Specular Color", r:0.23, g:0.23, b:0.23}
// bind intensity   {label:"Light Intensity", default:4, min:0, max:10}
// bind width       {label:"Width",  default: 8, min:0.1, max:15, step:0.1}
// bind height      {label:"Height", default: 8, min:0.1, max:15, step:0.1}
// bind roty        {label:"Rotation Y", default: 0, min:-0.5, max:0.5, step:0.001}
// bind rotz        {label:"Rotation Z", default: 0, min:0, max:1, step:0.001}
// bind twoSided    {label:"Two-sided", default:false}
// bind clipless    {label:"Clipless Approximation", default:false}
// bind groundTruth {label:"Ground Truth", default:false}


uniform float roughness;
uniform vec3  dcolor;
uniform vec3  scolor;

uniform float intensity;
uniform float width;
uniform float height;
uniform float roty;
uniform float rotz;
uniform float targetu;
uniform float targetv;

// obstacle shape
uniform float width_obstacle;
uniform float height_obstacle;
uniform float roty_obstacle;
uniform float rotz_obstacle;
uniform float posx;

bool twoSided = false;
uniform bool clipless;
uniform bool shadow_debug;
uniform bool shadow;
uniform bool second_obstacle;
uniform bool groundTruth;

uniform sampler2D ltc_1;
uniform sampler2D ltc_2;

uniform mat4  view;
uniform vec2  resolution;
uniform int   sampleCount;

const float LUT_SIZE  = 64.0;
const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
const float LUT_BIAS  = 0.5/LUT_SIZE;

const int   NUM_SAMPLES = 1;
const float pi = 3.14159265;

// Tracing and intersection
///////////////////////////

struct Ray
{
    vec3 origin;
    vec3 dir;
};

struct Rect
{
    vec3  center;
    vec3  dirx;
    vec3  diry;
    float halfx;
    float halfy;

    vec4  plane;
};

bool RayPlaneIntersect(Ray ray, vec4 plane, out float t)
{
    t = -dot(plane, vec4(ray.origin, 1.0))/dot(plane.xyz, ray.dir);
    return t > 0.0;
}

bool RayRectIntersect(Ray ray, Rect rect, out float t)
{
    bool intersect = RayPlaneIntersect(ray, rect.plane, t);
    if (intersect)
    {
        vec3 pos  = ray.origin + ray.dir*t;
        vec3 lpos = pos - rect.center;

        // rect 중심으로부터 거리를 잰다
        float x = dot(lpos, rect.dirx);
        float y = dot(lpos, rect.diry);

        // rect 중심의로부터 거리를 비교한다
        if (abs(x) > rect.halfx || abs(y) > rect.halfy)
            intersect = false;
    }

    return intersect;
}

// Camera functions
///////////////////

Ray GenerateCameraRay()
{
    Ray ray;

    // gl_FragCoord: coordinates of an input fragment in view(window) space
    // resolution: resolution of view space(window)
    vec2 xy = 2.0*gl_FragCoord.xy/resolution - vec2(1.0);

    ray.dir = normalize(vec3(xy, 2.0));

    float focalDistance = 2.0;
    float ft = focalDistance/ray.dir.z;
    vec3 pFocus = ray.dir*ft;

    ray.origin = vec3(0);
    ray.dir    = normalize(pFocus - ray.origin);

    // Apply camera transform
    ray.origin = (view*vec4(ray.origin, 1)).xyz;
    ray.dir    = (view*vec4(ray.dir,    0)).xyz;

    return ray;
}

vec3 mul(mat3 m, vec3 v)
{
    return m * v;
}

mat3 mul(mat3 m1, mat3 m2)
{
    return m1 * m2;
}

vec3 rotation_y(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) + v.z*sin(a);
    r.y =  v.y;
    r.z = -v.x*sin(a) + v.z*cos(a);
    return r;
}

vec3 rotation_z(vec3 v, float a)
{
    vec3 r;
    r.x =  v.x*cos(a) - v.y*sin(a);
    r.y =  v.x*sin(a) + v.y*cos(a);
    r.z =  v.z;
    return r;
}

vec3 rotation_yz(vec3 v, float ay, float az)
{
    return rotation_z(rotation_y(v, ay), az);
}

// Linearly Transformed Cosines
///////////////////////////////

vec3 IntegrateEdgeVec(vec3 v1, vec3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206*y)*y;
    float b = 3.4175940 + (4.1616724 + y)*y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5*inversesqrt(max(1.0 - x*x, 1e-7)) - v;

    return cross(v1, v2)*theta_sintheta;
}

float IntegrateEdge(vec3 v1, vec3 v2)
{
    return IntegrateEdgeVec(v1, v2).z;
}

void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4) impossible
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] =  L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3) impossible
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4
    {
        n = 4;
    }

    if (n == 3)
        L[3] = L[0];
    if (n == 4)
        L[4] = L[0];
}


vec3 LTC_Evaluate(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
{
    // construct orthonormal basis around N
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    // rotate area light in (T1, T2, N) basis
    Minv = mul(Minv, transpose(mat3(T1, T2, N)));

    // polygon (allocate 5 vertices for clipping)
    vec3 L[points.length()+1];
    for(int i = 0; i < points.length(); i++)
    {
        L[i] = mul(Minv, points[i] - P);
    }

    // integrate
    float sum = 0.0;

    if (clipless)
    {
        vec3 dir = points[0].xyz - P;
        vec3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
        bool behind = (dot(dir, lightNormal) < 0.0);

        L[0] = normalize(L[0]);
        L[1] = normalize(L[1]);
        L[2] = normalize(L[2]);
        L[3] = normalize(L[3]);

        vec3 vsum = vec3(0.0);

        vsum += IntegrateEdgeVec(L[0], L[1]);
        vsum += IntegrateEdgeVec(L[1], L[2]);
        vsum += IntegrateEdgeVec(L[2], L[3]);
        vsum += IntegrateEdgeVec(L[3], L[0]);

        float len = length(vsum);
        float z = vsum.z/len;

        if (behind)
            z = -z;

        vec2 uv = vec2(z*0.5 + 0.5, len);
        uv = uv*LUT_SCALE + LUT_BIAS;

        float scale = texture(ltc_2, uv).w;

        sum = len*scale;

        if (behind && !twoSided)
            sum = 0.0;
    }
    else
    {
        int n;
        ClipQuadToHorizon(L, n);

        if (n == 0)
            return vec3(0, 0, 0);
        // project onto sphere
        for(int i = 0; i < n; i++)
        {
            L[i] = normalize(L[i]);
        }

        // integrate
        for(int i = 0; i < n-1; i++)
        {
            sum += IntegrateEdge(L[i], L[i+1]);
        }
        sum += IntegrateEdge(L[n-1], L[0]);

        sum = twoSided ? abs(sum) : max(0.0, sum);
    }

    vec3 Lo_i = vec3(sum, sum, sum);

    return Lo_i;
}

vec3 LTC_Obstacle_Evaluate(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[8], int num_vertex, bool twoSided)
{
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    Minv = mul(Minv, transpose(mat3(T1, T2, N)));

    vec3 L[points.length()+1];
    for(int i = 0; i < num_vertex; i++) //
    {
        L[i] = mul(Minv, points[i] - P);
    }

    // integrate
    float sum = 0.0;

    if (num_vertex == 0)
        return vec3(0, 0, 0);
    // project onto sphere
    for(int i = 0; i < num_vertex; i++)
    {
        L[i] = normalize(L[i]);
    }

    // integrate
    for(int i = 0; i < num_vertex-1; i++)
    {
        sum += IntegrateEdge(L[i], L[i+1]);
    }
    sum += IntegrateEdge(L[num_vertex-1], L[0]);

    sum = twoSided ? abs(sum) : max(0.0, sum);

    vec3 Lo_i = vec3(sum, sum, sum);

    return Lo_i;
}


// TODO: replace this
float rand(vec2 co)
{
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float Halton(int index, float base)
{
    float result = 0.0;
    float f = 1.0/base;
    float i = float(index);
    for (int x = 0; x < 8; x++)
    {
        if (i <= 0.0) break;

        result += f*mod(i, base);
        i = floor(i/base);
        f = f/base;
    }

    return result;
}

void Halton2D(out vec2 s[NUM_SAMPLES], int offset)
{
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        s[i].x = Halton(i + offset, 2.0);
        s[i].y = Halton(i + offset, 3.0);
    }
}

vec3 LTC_GroundTruth(
    vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided, float u1, float u2)
{
    vec3 T1, T2;
    T1 = normalize(V - N*dot(V, N));
    T2 = cross(N, T1);

    Minv = mul(Minv, transpose(mat3(T1, T2, N)));

    vec3 L[4];
    for(int i = 0; i < 4; i++)
    {
        L[i] = mul(Minv, points[i] - P);
    }
    
    // integrate
    float sum = 0.0;

    float w = sqrt(dot(L[2]-L[1], L[2]-L[1]));
    float h = sqrt(dot(L[3]-L[2], L[3]-L[2]));

    float lightArea = w*h;

    // light sample
    {
        float rad = sqrt(u1);
        float phi = 2.0*pi*u2;
        float x = rad*cos(phi);
        float y = rad*sin(phi);
        vec3 dir = vec3(x, y, sqrt(1.0 - u1));

        Ray ray;
        ray.origin = P;
        ray.dir = dir;

        Rect rect;
        rect.dirx = rotation_yz(vec3(1, 0, 0), roty*2.0*pi, rotz*2.0*pi);
        rect.diry = rotation_yz(vec3(0, 1, 0), roty*2.0*pi, rotz*2.0*pi);

        rect.center = vec3(0, 6, 32);
        rect.halfx  = 0.5*width;
        rect.halfy  = 0.5*height;

        vec3 rectNormal = cross(rect.dirx, rect.diry);
        rect.plane = vec4(rectNormal, -dot(rectNormal, rect.center));

        float distToRect;
        bool hitLight = RayRectIntersect(ray, rect, distToRect);

        float cosTheta = max(dir.z, 0.0);
        float brdf = 1.0/pi;
        float pdfBRDF = cosTheta/pi;

        float c2 = max(dot(dir, rectNormal), 0.0);
        float solidAngle = max(c2/distToRect/distToRect, 1e-7);
        float pdfLight = 1.0/solidAngle/lightArea;
        
        if (hitLight)
        {
            sum += brdf*cosTheta/(pdfBRDF + pdfLight);
        }
    }

    vec3 Lo_i = vec3(sum, sum, sum);

    return Lo_i;
}

// Scene helpers
////////////////

void InitRect(out Rect rect)
{
    rect.dirx = rotation_yz(vec3(1, 0, 0), roty*2.0*pi, rotz*2.0*pi);
    rect.diry = rotation_yz(vec3(0, 1, 0), roty*2.0*pi, rotz*2.0*pi);

    rect.center = vec3(0, 6, 32);
    rect.halfx  = 0.5*width;
    rect.halfy  = 0.5*height;

    vec3 rectNormal = cross(rect.dirx, rect.diry);
    rect.plane = vec4(rectNormal, -dot(rectNormal, rect.center));
}

void InitRectPoints(Rect rect, out vec3 points[4])
{
    vec3 ex = rect.halfx*rect.dirx;
    vec3 ey = rect.halfy*rect.diry;

    points[0] = rect.center - ex - ey;
    points[1] = rect.center + ex - ey;
    points[2] = rect.center + ex + ey;
    points[3] = rect.center - ex + ey;
}

void InitObstacle(out Rect square, vec3 center, float w, float h)
{
    square.dirx = rotation_yz(vec3(1, 0, 0), roty_obstacle*2.0*pi, rotz_obstacle*2.0*pi); // should be an unit vector
    square.diry = rotation_yz(vec3(0, 1, 0), roty_obstacle*2.0*pi, rotz_obstacle*2.0*pi);

    square.center = center;
    square.halfx  = 0.5*w;
    square.halfy  = 0.5*h;

    vec3 rectNormal = cross(square.dirx, square.diry);
    square.plane = vec4(rectNormal, -dot(rectNormal, square.center));   
}

// Shadow Geometry Helpers
///////////////////

// defined by a point on a plane, and two basis vectors for UV coordinate
struct Plane{
    vec3 pt;
    vec3 eu;
    vec3 ev;
};

// get plane defined by 3 points 
void getPlane(out Plane plane, vec3 p1, vec3 p2, vec3 p3){
    plane.pt = p1;
    plane.eu = p2 - p1;
    plane.ev = p3 - p1;
    // making orthonormal bases might help vertex simplification and performance
}

// perspective projection to a plane, get UV coordinate in plane
vec2 projPersp(Plane plane, vec3 viewPt, vec3 projXYZ){
    // some linear algebra...
    // Planept + cu*Eu + cv*Ev = ProjectionPt =  Viewpt + t*Raydir
    // cu*Eu + cv*Ev - t*Raydir = Viewpt - Planept
    // [Eu Ev Raydir]*([cu cv -t]') = (Viewpt - Planept)
    // [cu cv -t] = inv([Eu Ev Raydir])*(Viewpt - Planept)

    mat3 bases = mat3(plane.eu, plane.ev, projXYZ - viewPt);
    mat3 invBases = inverse(bases);
    vec3 coords = invBases*(viewPt - plane.pt);
    vec2 projUV = vec2(coords[0], coords[1]);

    // DEBUG: a fake statement!
    vec2 zero = vec2(targetu, targetv);
    projUV + 0.0*zero;

    return projUV;
}

// get UV coordinate of a point on a plane
vec2 planePtUV(Plane plane, vec3 ptXYZ){
    mat3 bases = mat3(plane.eu, plane.ev, cross(plane.eu, plane.ev));
    mat3 invBases = inverse(bases);
    vec3 coords = invBases*(ptXYZ - plane.pt);
    vec2 ptUV = vec2(coords[0], coords[1]);
    return ptUV;
}

// convert plane 2D coordinate (uv) to XYZ coordinate point
vec3 UVtoXYZ(Plane plane, vec2 uv){
    vec3 ptXYZ = plane.pt + uv[0]*plane.eu + uv[1]*plane.ev;
    return ptXYZ;
}


// check if P is inside polygon edge ep1->ep2
bool inside(vec2 P, vec2 ep1, vec2 ep2){
    vec3 v1 = vec3(ep2[0]-ep1[0], ep2[1]-ep1[1], 0.0);
    vec3 v2 = vec3(P[0]-ep1[0], P[1]-ep1[1], 0.0);
    vec3 res = cross(v1, v2);
    return res[2] > 0.0;
}

// get cross intersection point of two straight lines
vec2 cross_pt(vec2 p1, vec2 p2, vec2 q1, vec2 q2){
    // p1 + t*Vp = q1 + s*Vq
    // t*Vp - s*Vq = q1 - p1
    // [t -s] = inv([Vp Vq])*(q1 - p1)
    vec2 Vp = p2 - p1;
    vec2 Vq = q2 - q1;
    vec2 coords = inverse(mat2(Vp, Vq))*(q1 - p1);
    return p1+coords[0]*Vp;
}

// get clipped obstacle. all lists are couter-clockwise
void clipObstacleUV(out vec2 outputList[8], out int num_vertex, vec2 light[4], vec2 obstacle[4]){
    // Use Sutherland-Hodgeman algorithm to compute intersection of CONVEX polygons
    vec2 ep1, ep2;
    vec2 inputList[8];
    vec2 S, E;
    
    num_vertex = 4; // number of vertices in current clipped polygon
    // outputList = obstacle;
    for(int v = 0; v < num_vertex; v++){
        outputList[v] = obstacle[v];
    }
    int num_clip = 4; // number of clip edges (light)
    for(int e = 0; e < num_clip; e++){

        // get two edge points
        ep1 = light[e];
        ep2 = light[(e+1)%num_clip];

        // get new vertices
        inputList = outputList;
        int vertex_cnt = 0;
        S = inputList[num_vertex - 1]; // previous vertex
        for(int v = 0; v < num_vertex; v++){
            // traverse polygon to add / skip / create vertices
            E = inputList[v]; // current vertex
            if(inside(E, ep1, ep2)){
                if(!inside(S, ep1, ep2)){
                    // intersection
                    outputList[vertex_cnt] = cross_pt(S, E, ep1, ep2);
                    vertex_cnt++;
                }
                outputList[vertex_cnt] = E; // E included
                vertex_cnt++;
            }
            else{
                if(inside(S, ep1, ep2)){
                    // intersection
                    outputList[vertex_cnt] = cross_pt(S, E, ep1, ep2);
                    vertex_cnt++;
                }
                // E excluded
            }
            S = E;
        }
        num_vertex = vertex_cnt;
    }
}

// perspective-project obstacle to light plane, and then clip it by light edges
void clipProjectObstacle(out vec3 result[8], out int num_vertex, vec3 light[4], vec3 obstacle[4], vec3 viewPt){

    Plane lightPlane;
    vec2 lightUV[4];
    vec2 obstacleUV[4];
    vec2 clippedUV[8];

    // perspective projection in light plane coordinates
    getPlane(lightPlane, light[0], light[1], light[2]);
    for(int i = 0; i < 4; i++){
        lightUV[i] = planePtUV(lightPlane, light[i]);
    }
    for(int i = 0; i < 4; i++){
        obstacleUV[i] = projPersp(lightPlane, viewPt, obstacle[i]);
    }

    // clipping by light edges
    clipObstacleUV(clippedUV, num_vertex, lightUV, obstacleUV);

    // recover from light plane coordinates
    for(int i = 0; i < num_vertex; i++){
        result[i] = UVtoXYZ(lightPlane, clippedUV[i]);
    }

}


void getDebugPoints(out vec3 noclipped[8], out vec3 clipped[8], out int num_vertex, vec3 light[4], vec3 obstacle[4], vec3 viewPt){
    Plane lightPlane;
    vec2 lightUV[4];
    vec2 obstacleUV[4];
    vec2 clippedUV[8];

    // perspective projection in light plane coordinates
    getPlane(lightPlane, light[0], light[1], light[2]);
    for(int i = 0; i < 4; i++){
        lightUV[i] = planePtUV(lightPlane, light[i]);
    }
    for(int i = 0; i < 4; i++){
        obstacleUV[i] = projPersp(lightPlane, viewPt, obstacle[i]);
    }

    // clipping by light edges
    clipObstacleUV(clippedUV, num_vertex, lightUV, obstacleUV);

    // recover from light plane coordinates
    for(int i = 0; i < num_vertex; i++){
        clipped[i] = UVtoXYZ(lightPlane, clippedUV[i]);
    }
    for(int i = 0; i < 4; i++){
        noclipped[i] = UVtoXYZ(lightPlane, obstacleUV[i]);
    }
}


// Shadow Debug Helpers
///////////////////////

bool camHitPoint(vec3 x0){
    Ray ray = GenerateCameraRay();
    vec3 x1 = ray.origin;
    vec3 x2 = x1 + ray.dir;
    // dist between line and point
    float thres = 0.1;
    float d = length(cross(x0 - x1, x0 - x2))/length(x1 - x2);
    if(d < thres){
        return true;
    }
    else{
        return false;
    }
}


bool camHitRay(Ray targetRay){
    Ray camRay = GenerateCameraRay();
    vec3 n = cross(camRay.dir, targetRay.dir);
    float d = abs(dot(n, camRay.origin - targetRay.origin))/length(n);
    float thres = 0.05;
    if(d < thres){
        return true;
    }
    else{
        return false;
    }
}

bool camHitSegment(vec3 p1, vec3 p2){
    Ray camRay = GenerateCameraRay();
    Ray targetRay;
    targetRay.origin = p1;
    targetRay.dir = p2 - p1;
    vec3 n = cross(targetRay.dir, camRay.dir);
    vec3 n2 = cross(camRay.dir, n);
    float d = abs(dot(n, targetRay.origin - camRay.origin))/length(n);
    float t = dot(camRay.origin - targetRay.origin, n2)/dot(targetRay.dir, n2);
    float thres = 0.05;
    if(d<thres && t > 0.0 && t < 1.0){
        return true;
    }
    else{
        return false;
    }
}


// Misc. helpers
////////////////

float saturate(float v)
{
    return clamp(v, 0.0, 1.0);
}

vec3 PowVec3(vec3 v, float p)
{
    return vec3(pow(v.x, p), pow(v.y, p), pow(v.z, p));
}

const float gamma = 2.2;
vec3 ToLinear(vec3 v) { return PowVec3(v, gamma); }

out vec4 FragColor;

void main()
{
    // Initialize rectangular light's shape and position
    Rect rect;
    InitRect(rect);
    vec3 points[4];
    InitRectPoints(rect, points);

    // Initialize a square obstacle
    Rect obstacle;
    InitObstacle(obstacle, vec3(posx,7,28), width_obstacle, height_obstacle);
    vec3 obstaclePoints[4];
    InitRectPoints(obstacle, obstaclePoints);

    Rect obstacle2;
    InitObstacle(obstacle2, vec3(-posx,7,28), width_obstacle, height_obstacle);
    vec3 obstaclePoints2[4];
    InitRectPoints(obstacle2, obstaclePoints2);

    // The floor was defined by its normal vector
    // In the CG field, the z-coordinate is interpreted as the depth w.r.t the camera
    // y-coordinate: upside
    vec4 floorPlane = vec4(0, 1, 0, 0);

    vec3 lcol = vec3(intensity); // light color
    vec3 dcol = ToLinear(dcolor); // diffuse color
    vec3 scol = ToLinear(scolor); // specular color

    vec3 col = vec3(1);

    Ray ray = GenerateCameraRay();

    // Ray-floor intersection
    float distToFloor;
    bool hitFloor = RayPlaneIntersect(ray, floorPlane, distToFloor);

    // if the fragment is floor, color is obtained by rendering Eq.
    if (hitFloor)
    {
        // ray-floor intersenction position
        vec3 pos = ray.origin + ray.dir*distToFloor;

        // Normal vector & Viewing vector
        vec3 N = floorPlane.xyz; // since the floor was defined by its normal vector
        vec3 V = -ray.dir;

        // viewing direction
        float ndotv = saturate(dot(N, V));

        // LTCs are stored in Look-up table(LUT)
        // LTCs are located by roughness and viewing direction 
        vec2 uv = vec2(roughness, sqrt(1.0 - ndotv));

        // LTCs are stored in Look-up table
        // calculate look-up table index
        uv = uv*LUT_SCALE + LUT_BIAS;

        // Table for floor material
        vec4 t1 = texture(ltc_1, uv);
        // Table for light source (불확실함)
        vec4 t2 = texture(ltc_2, uv);

        // M^(-1)
        mat3 Minv = mat3(
            vec3(t1.x, 0, t1.y),
            vec3(  0,  1,    0),
            vec3(t1.z, 0, t1.w)
        );

        vec3 spec;
        vec3 diff;
        
        if (groundTruth)
        {
            // random sampling
            vec2 seq[NUM_SAMPLES];
            Halton2D(seq, sampleCount);

            float u1 = rand(gl_FragCoord.xy*0.01);
            float u2 = rand(gl_FragCoord.yx*0.01);

            u1 = fract(u1 + seq[0].x);
            u2 = fract(u2 + seq[0].y);

            spec = LTC_GroundTruth(N, V, pos, Minv, points, twoSided, u1, u2);
            diff = LTC_GroundTruth(N, V, pos, mat3(1), points, twoSided, u1, u2);
        }
        else
        {
            spec = LTC_Evaluate(N, V, pos, Minv, points, twoSided);

            // BRDF shadowing and Fresnel
            // 뭔진 잘 모르겠다
            // 반사광의 밝기를 조절해주는 듯 하다
            
    
            // mat3(1) = 3*3 identity matrix
            // identity matrix에 대한 LTC는 바로 그냥 cosine이다.
            // 즉 이에 대한 illumination을 계산한다는 것은, 
            // perfect lambertian illumination을 계산한다는 것을 뜻한다
            diff = LTC_Evaluate(N, V, pos, mat3(1), points, twoSided);
            
            if(shadow)
            {
                // Obstacle LTC Evaluate
                int num_vertex;
                vec3 clipped_points[8];
                clipProjectObstacle(clipped_points, num_vertex, points, obstaclePoints, pos);
        
                vec3 obstacle_spec = LTC_Obstacle_Evaluate(N, V, pos, Minv, clipped_points, num_vertex, twoSided);
                spec -= obstacle_spec;
                vec3 obstacle_diff = LTC_Obstacle_Evaluate(N, V, pos, mat3(1), clipped_points, num_vertex, twoSided);
                diff -= obstacle_diff;
    
                if (second_obstacle)
                {
                    // Obstacle LTC Evaluate
                    int num_vertex2;
                    vec3 clipped_points2[8];
                    clipProjectObstacle(clipped_points2, num_vertex2, points, obstaclePoints2, pos);
            
                    vec3 obstacle_spec2 = LTC_Obstacle_Evaluate(N, V, pos, Minv, clipped_points2, num_vertex2, twoSided);
                    spec -= obstacle_spec2;
                    vec3 obstacle_diff2 = LTC_Obstacle_Evaluate(N, V, pos, mat3(1), clipped_points2, num_vertex2, twoSided);
                    diff -= obstacle_diff2;
                }
            }
        }

        spec *= scol*t2.x + (1.0 - scol)*t2.y;
        col = lcol*(spec + dcol*diff);
    }

    // Ray-obstacle intersection
    float distToObstacle;
    bool hitObstacle = RayRectIntersect(ray, obstacle, distToObstacle);

    // if the fragment is light source, color is light color
    float distToRect;
    bool hitLight = RayRectIntersect(ray, rect, distToRect);

    if (hitLight)
    {
        if ((distToRect < distToFloor) || !hitFloor)
        {
            col = lcol;
        }
    }
    
    if (hitObstacle)
    {
        // obstacle을 그릴 조건. 
        if (((distToObstacle < distToFloor) || !hitFloor) 
            && ((distToObstacle < distToRect) || !hitLight))
        {
            // Dummy color, just for now
            col = vec3(0);
        }
    }

    if (second_obstacle)
    {
        float distToObstacle2;
        bool hitObstacle2 = RayRectIntersect(ray, obstacle2, distToObstacle2);
        if (hitObstacle2)
        {
            // obstacle을 그릴 조건. 
            if (((distToObstacle2 < distToFloor) || !hitFloor) 
                && ((distToObstacle2 < distToRect) || !hitLight))
            {
                // Dummy color, just for now
                col = vec3(0);
            }
        }
    }

    if(shadow_debug){

        // target view point
        float targetScale = 15.0;
        vec3 viewPt = vec3(0.0+targetScale*targetu, 0.0, 10.0+targetScale*targetv);

        // get clipped vertices
        InitRectPoints(rect, points);
        int num_vertex;

        vec3 clipped_points[8];
        vec3 noclipped_points[8];
        int nv;
        getDebugPoints(noclipped_points, clipped_points, nv, points, obstaclePoints, viewPt);
        

        // display view ray segments
        for(int i=0; i<4; i++){
            bool hitSegment = camHitSegment(viewPt, noclipped_points[i]);
            if(hitSegment){
                col = vec3(0,0.2,0);
            }
        }
        // display unclipped polygon
        for(int i=0; i<4; i++){
            vec3 p1 = noclipped_points[i];
            vec3 p2 = noclipped_points[(i+1)%4];
            bool hitViewPt = camHitSegment(p1, p2);
            if(hitViewPt){
                col = vec3(0,0.2,0);
            }
        }
        // display clipped polygon
        for(int i=0; i<nv; i++){
            vec3 p1 = clipped_points[i];
            vec3 p2 = clipped_points[(i+1)%nv];
            bool hitViewPt = camHitSegment(p1, p2);
            if(hitViewPt){
                col = vec3(0,0.5,0.8);
            }
        }

        /*
        // display clipped points
        for(int i=0; i<4; i++){
            vec3 target = noclipped_points[i];
            bool hitViewPt = camHitPoint(target);
            if(hitViewPt){
                col = vec3(1,0,0);
            }
        }
        for(int i=0; i<nv; i++){
            vec3 target = clipped_points[i];
            bool hitViewPt = camHitPoint(target);
            if(hitViewPt){
                col = vec3(0,0,1);
            }
        }
        */
        // display target view point
        bool hitViewPt = camHitPoint(viewPt);
        if(hitViewPt){
            col = vec3(0,0,1);
        }

    }
    
    
    // 아무것도 없는 곳에는 col = vec3(0) 이다

    FragColor = vec4(col, 1.0);
}