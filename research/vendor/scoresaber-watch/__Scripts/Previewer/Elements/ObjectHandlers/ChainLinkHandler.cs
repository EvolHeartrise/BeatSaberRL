using UnityEngine;

public class ChainLinkHandler : MonoBehaviour
{
    public AudioSource audioSource;
    public bool Visible { get; private set; }

    [SerializeField] private MeshRenderer meshRenderer;
    [SerializeField] private MeshRenderer dotMeshRenderer;
    [SerializeField] private MeshRenderer outlineRenderer;

    private bool outline;
    private Color outlineColor;

    private Material baseDotMaterial;

    private MaterialPropertyBlock outlineProperties;


    public Material BaseDotMaterial
    {
        get
        {
            if(baseDotMaterial == null) baseDotMaterial = dotMeshRenderer.sharedMaterial;
            return baseDotMaterial;
        }
    }


    public void SetSharedMaterials(Material linkMaterial, Material dotMaterial)
    {
        meshRenderer.sharedMaterial = linkMaterial;
        dotMeshRenderer.sharedMaterial = dotMaterial;
        ClearProperties();
    }


    public void SetCustomMaterials(Material linkMaterial)
    {
        meshRenderer.sharedMaterial = linkMaterial;
        dotMeshRenderer.sharedMaterial = BaseDotMaterial;
        ClearProperties();
    }


    public void SetProperties(MaterialPropertyBlock properties)
    {
        meshRenderer.SetPropertyBlock(properties);
    }


    public void SetDotProperties(MaterialPropertyBlock properties)
    {
        dotMeshRenderer.SetPropertyBlock(properties);
    }


    private void ClearProperties()
    {
        meshRenderer.SetPropertyBlock(null);
        dotMeshRenderer.SetPropertyBlock(null);
    }


    private void Awake()
    {
        baseDotMaterial = dotMeshRenderer.sharedMaterial;
    }


    public void SetOutline(bool useOutline)
    {
        outline = useOutline;
        outlineRenderer.gameObject.SetActive(useOutline);
    }


    public void SetOutline(bool useOutline, Color color)
    {
        outline = useOutline;
        if(outline)
        {
            outlineColor = color;

            if(outlineProperties == null)
            {
                outlineProperties = new MaterialPropertyBlock();
            }
            outlineProperties.SetColor("_BaseColor", outlineColor);

            outlineRenderer.gameObject.SetActive(true);
            outlineRenderer.SetPropertyBlock(outlineProperties);
        }
        else
        {
            outlineRenderer.gameObject.SetActive(false);
        }
    }


    public void DisableVisual()
    {
        if(!Visible) return;

        dotMeshRenderer.gameObject.SetActive(false);
        outlineRenderer.gameObject.SetActive(false);
        meshRenderer.enabled = false;
        Visible = false;
    }


    public void EnableVisual()
    {
        if(Visible) return;

        SetOutline(outline);
        dotMeshRenderer.gameObject.SetActive(true);
        meshRenderer.enabled = true;
        Visible = true;
    }
}