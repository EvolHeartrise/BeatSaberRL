using UnityEngine;

public class ArcHandler : MonoBehaviour
{
    private static readonly int textureOffsetID = Shader.PropertyToID("_TextureOffset");
    private static readonly int fadeStartPointID = Shader.PropertyToID("_FadeStartPoint");
    private static readonly int baseColorID = Shader.PropertyToID("_BaseColor");

    [SerializeField] private LineRenderer lineRenderer;

    private MaterialPropertyBlock materialProperties;
    private Color lastBaseColor;
    private float lastTextureOffset;
    private float lastCloseFadeDist;
    private bool propertiesDirty = true;


    private static bool ColorValuesEqual(Color a, Color b)
    {
        return a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a;
    }


    public void SetArcPoints(Vector3[] newPoints)
    {
        SetArcPoints(newPoints, newPoints.Length);
    }


    public void SetArcPoints(Vector3[] newPoints, int pointCount)
    {
        lineRenderer.positionCount = pointCount;
        if(pointCount == newPoints.Length)
        {
            lineRenderer.SetPositions(newPoints);
            return;
        }

        for(int i = 0; i < pointCount; i++)
        {
            lineRenderer.SetPosition(i, newPoints[i]);
        }
    }


    public void SetGradient(float curveLength, float endFadeStart, float endFadeLength)
    {
        //Sets the alpha gradient of the linerenderer to make the end fades consistent
        //Needed since gradients are based on percentage of length, not actual distance,
        //so longer arcs would have a longer fade at the end without this
        float fadeStart = endFadeStart / curveLength;
        float fadeEnd = endFadeLength / curveLength;

        Gradient gradient = new Gradient();
        gradient.SetKeys(
            new GradientColorKey[]
            {
                new GradientColorKey(Color.white, 0f)
            },
            new GradientAlphaKey[]
            {
                new GradientAlphaKey(0f, fadeStart),
                new GradientAlphaKey(1f, fadeEnd),
                new GradientAlphaKey(1f, 1f - fadeEnd),
                new GradientAlphaKey(0f, 1f - fadeStart)
            }
        );
        lineRenderer.colorGradient = gradient;
    }


    public void SetWidth(float width)
    {
        lineRenderer.startWidth = width;
        lineRenderer.endWidth = width;
    }


    public void SetMaterial(Material newMaterial, MaterialPropertyBlock properties)
    {
        lineRenderer.sharedMaterial = newMaterial;
        lineRenderer.SetPropertyBlock(properties);

        //This creates a proper copy of the material property block
        //Allows us to change properties on this arc without worrying about reference types
        if(materialProperties == null) materialProperties = new MaterialPropertyBlock();
        lineRenderer.GetPropertyBlock(materialProperties);
        lastBaseColor = materialProperties.GetColor(baseColorID);
        propertiesDirty = true;
    }


    public void SetProperties(float alpha, float textureOffset, Color? customColor, float closeFadeDist)
    {
        Color color = customColor ?? lastBaseColor;
        color.a = alpha;

        if(!propertiesDirty
            && lastTextureOffset == textureOffset
            && lastCloseFadeDist == closeFadeDist
            && ColorValuesEqual(lastBaseColor, color))
        {
            return;
        }

        lastTextureOffset = textureOffset;
        lastCloseFadeDist = closeFadeDist;
        lastBaseColor = color;
        propertiesDirty = false;

        materialProperties.SetFloat(textureOffsetID, textureOffset);
        materialProperties.SetFloat(fadeStartPointID, closeFadeDist);
        materialProperties.SetColor(baseColorID, color);

        lineRenderer.SetPropertyBlock(materialProperties);
    }
}