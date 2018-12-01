// bind roughness   {label:"Roughness", default:0.25, min:0.01, max:1, step:0.001}
// bind dcolor      {label:"Diffuse Color",  r:1.0, g:1.0, b:1.0}
// bind scolor      {label:"Specular Color", r:0.23, g:0.23, b:0.23}
// bind intensity   {label:"Light Intensity", default:4, min:0, max:10}
// bind width       {label:"Width",  default: 8, min:0.1, max:15, step:0.1}
// bind height      {label:"Height", default: 8, min:0.1, max:15, step:0.1}
// bind roty        {label:"Rotation Y", default: 0, min:0, max:1, step:0.001}
// bind rotz        {label:"Rotation Z", default: 0, min:0, max:1, step:0.001}
// bind twoSided    {label:"Two-sided", default:false}
// bind clipless    {label:"Clipless Approximation", default:false}

uniform float roughness;
uniform vec3  dcolor;
uniform vec3  scolor;

uniform float intensity;
uniform float width;
uniform float height;
uniform float roty;
uniform float rotz;

bool twoSided = true;
uniform bool clipless;

uniform sampler2D ltc_1;
uniform sampler2D ltc_2;

uniform mat4  view;
uniform vec2  resolution;
uniform int   sampleCount;

const float LUT_SIZE  = 64.0;
const float LUT_SCALE = (LUT_SIZE - 1.0)/LUT_SIZE;
const float LUT_BIAS  = 0.5/LUT_SIZE;

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
        L[0] = normalize(L[0]);
        L[1] = normalize(L[1]);
        L[2] = normalize(L[2]);
        L[3] = normalize(L[3]);
        L[4] = normalize(L[4]);

        // integrate
        sum += IntegrateEdge(L[0], L[1]);
        sum += IntegrateEdge(L[1], L[2]);
        sum += IntegrateEdge(L[2], L[3]);
        if (n >= 4)
            sum += IntegrateEdge(L[3], L[4]);
        if (n == 5)
            sum += IntegrateEdge(L[4], L[0]);

        sum = twoSided ? abs(sum) : max(0.0, sum);
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
    square.dirx = rotation_yz(vec3(1, 0, 0), roty*2.0*pi, rotz*2.0*pi); // should be an unit vector
    square.diry = rotation_yz(vec3(0, 1, 0), roty*2.0*pi, rotz*2.0*pi);

    square.center = center;
    square.halfx  = 0.5*w;
    square.halfy  = 0.5*h;

    vec3 rectNormal = cross(square.dirx, square.diry);
    square.plane = vec4(rectNormal, -dot(rectNormal, square.center));   
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
    InitObstacle(obstacle, vec3(2,2,28), 4.0, 4.0);
    vec3 obstaclePoints[4];
    InitRectPoints(obstacle, obstaclePoints);

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

        // points: each vertex of the polyonal light
        // LTC_Evaluate returns vec3 =  three color coordinates
        vec3 spec = LTC_Evaluate(N, V, pos, Minv, points, twoSided);
        // BRDF shadowing and Fresnel
        // 뭔진 잘 모르겠다
        // 반사광의 밝기를 조절해주는 듯 하다
        spec *= scol*t2.x + (1.0 - scol)*t2.y;

        // mat3(1) = 3*3 identity matrix
        // identity matrix에 대한 LTC는 바로 그냥 cosine이다.
        // 즉 이에 대한 illumination을 계산한다는 것은, 
        // perfect lambertian illumination을 계산한다는 것을 뜻한다
        vec3 diff = LTC_Evaluate(N, V, pos, mat3(1), points, twoSided);

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
    
    // 아무것도 없는 곳에는 col = vec3(0) 이다

    FragColor = vec4(col, 1.0);
}