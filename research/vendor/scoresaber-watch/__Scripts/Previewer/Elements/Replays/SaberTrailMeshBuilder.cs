using System.Collections.Generic;
using UnityEngine;

public class SaberTrailMeshBuilder : MonoBehaviour
{
    [SerializeField] private MeshFilter meshFilter;

    [Space]
    [SerializeField] private bool isRightHand;
    [SerializeField] private Vector3 tipOffset;

    private Mesh mesh;
    private Vector3[] vertices;
    private Vector2[] uvs;
    private int[] triangles;
    private int currentSegmentCount = -1;
    private bool meshAssigned;
    private float trailLifetime;
    private float trailWidth;
    private int trailSegmentCount = 10;


    private void EnsureMeshAssigned()
    {
        if(meshAssigned)
        {
            return;
        }

        mesh.MarkDynamic();
        meshFilter.mesh = mesh;
        meshAssigned = true;
    }


    private bool EnsureBuffers(int segmentCount)
    {
        if(segmentCount == currentSegmentCount)
        {
            return false;
        }

        currentSegmentCount = segmentCount;
        mesh.Clear();

        vertices = new Vector3[segmentCount * 2];
        uvs = new Vector2[vertices.Length];

        int faceCount = segmentCount - 1;
        int triangleCount = faceCount * 2;
        triangles = new int[triangleCount * 3];

        for(int i = 0; i < segmentCount; i++)
        {
            int handleIndex = i * 2;
            int tipIndex = handleIndex + 1;

            //UVs don't reach fully to the top/bottom because there're weird artifacts
            //for some reason
            float uvX = (float)i / segmentCount;
            uvs[handleIndex] = new Vector2(uvX, 0f);
            uvs[tipIndex] = new Vector2(uvX, 1f);

            if(i < segmentCount - 1)
            {
                //Add triangles linking to the next segment
                int bottomIndex = handleIndex * 3;
                triangles[bottomIndex] = handleIndex;
                triangles[bottomIndex + 1] = handleIndex + 1;
                triangles[bottomIndex + 2] = handleIndex + 2;

                int topIndex = tipIndex * 3;
                triangles[topIndex] = tipIndex;
                triangles[topIndex + 1] = tipIndex + 1;
                triangles[topIndex + 2] = tipIndex + 2;
            }
        }

        return true;
    }


    private void UpdateSettings(string setting)
    {
        if(!SettingsManager.Loaded || (setting != "all" && setting != "sabertraillength" && setting != "sabertrailwidth" && setting != "sabertrailsegments"))
        {
            return;
        }

        trailLifetime = Mathf.Clamp01(SettingsManager.GetFloat("sabertraillength"));
        trailWidth = Mathf.Clamp01(SettingsManager.GetFloat("sabertrailwidth"));
        trailSegmentCount = Mathf.Clamp(SettingsManager.GetInt("sabertrailsegments"), 10, 100);
    }


    public void SetFrames(List<ReplayFrame> frames, int startIndex)
    {
        float segmentLength = trailLifetime / trailSegmentCount;

        EnsureMeshAssigned();
        bool meshStructureChanged = EnsureBuffers(trailSegmentCount);
        Matrix4x4 worldToLocal = transform.worldToLocalMatrix;

        int frameIndex = startIndex;
        for(int i = 0; i < trailSegmentCount; i++)
        {
            float timeDifference = segmentLength * i;
            float segmentTime = Mathf.Max(TimeManager.CurrentTime - timeDifference, 0f);

            //Make sure the first segment always lines up with the last frame
            if(i > 0)
            {
                //Find the last frame used for this segment
                while(frames[frameIndex].Time > segmentTime && frameIndex > 0)
                {
                    frameIndex--;
                }
            }

            int handleIndex = i * 2;
            int tipIndex = handleIndex + 1;

            bool useNextFrame = frameIndex + 1 < frames.Count;
            ReplayFrame frame = frames[frameIndex];
            ReplayFrame nextFrame = useNextFrame ? frames[frameIndex + 1] : frame;

            float frameDifference = Mathf.Max(nextFrame.Time - frame.Time, 0.001f);
            float frameProgress = segmentTime - frame.Time;
            float t = frameProgress / frameDifference;

            Vector3 currentPosition = isRightHand ? frame.GetRightSaberPos() : frame.GetLeftSaberPos();
            Quaternion currentRotation = isRightHand ? frame.GetRightSaberRot() : frame.GetLeftSaberRot();
            Vector3 nextPosition = isRightHand ? nextFrame.GetRightSaberPos() : nextFrame.GetLeftSaberPos();
            Quaternion nextRotation = isRightHand ? nextFrame.GetRightSaberRot() : nextFrame.GetLeftSaberRot();

            Vector3 saberPosition = Vector3.Lerp(currentPosition, nextPosition, t);
            Quaternion saberRotation = Quaternion.Lerp(currentRotation, nextRotation, t);

            saberPosition.z -= ObjectManager.PlayerCutPlaneDistance;
            saberPosition = worldToLocal.MultiplyPoint3x4(saberPosition);

            Vector3 tipPoint = worldToLocal.MultiplyVector(saberRotation * tipOffset);

            Vector3 tipDirection = tipPoint.normalized;
            Vector3 handlePoint = tipPoint - (tipDirection * trailWidth);

            vertices[handleIndex] = saberPosition + handlePoint;
            vertices[tipIndex] = saberPosition + tipPoint;
        }

        mesh.SetVertices(vertices);

        if(meshStructureChanged)
        {
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(triangles, 0);
        }
    }


    private void OnEnable()
    {
        SettingsManager.OnSettingsUpdated += UpdateSettings;
        UpdateSettings("all");
    }


    private void OnDisable()
    {
        SettingsManager.OnSettingsUpdated -= UpdateSettings;
    }


    private void Awake()
    {
        mesh = new Mesh();
    }
}