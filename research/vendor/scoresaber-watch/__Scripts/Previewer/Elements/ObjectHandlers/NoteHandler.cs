using UnityEngine;

public class NoteHandler : MonoBehaviour
{
    public AudioSource audioSource;
    public bool Visible { get; private set; }

    [SerializeField] private MeshRenderer meshRenderer;
    [SerializeField] private MeshFilter meshFilter;
    [SerializeField] private MeshRenderer arrowMeshRenderer;
    [SerializeField] private MeshRenderer dotMeshRenderer;

    [SerializeField] private MeshRenderer outlineRenderer;
    [SerializeField] private MeshFilter outlineMeshFilter;

    private bool isArrow;
    private bool outline;
    private Color outlineColor;

    private Material baseArrowMaterial;
    private Material baseDotMaterial;

    private MaterialPropertyBlock outlineProperties;


    public Material BaseArrowMaterial
    {
        get
        {
            if(baseArrowMaterial == null) baseArrowMaterial = arrowMeshRenderer.sharedMaterial;
            return baseArrowMaterial;
        }
    }


    public Material BaseDotMaterial
    {
        get
        {
            if(baseDotMaterial == null) baseDotMaterial = dotMeshRenderer.sharedMaterial;
            return baseDotMaterial;
        }
    }


    public void SetMesh(Mesh newMesh)
    {
        meshFilter.mesh = newMesh;
        outlineMeshFilter.mesh = newMesh;
    }


    public void SetSharedMaterials(Material noteMaterial, Material arrowMaterial, Material dotMaterial)
    {
        meshRenderer.sharedMaterial = noteMaterial;
        arrowMeshRenderer.sharedMaterial = arrowMaterial;
        dotMeshRenderer.sharedMaterial = dotMaterial;
        ClearProperties();
    }


    public void SetCustomMaterials(Material noteMaterial)
    {
        meshRenderer.sharedMaterial = noteMaterial;
        arrowMeshRenderer.sharedMaterial = BaseArrowMaterial;
        dotMeshRenderer.sharedMaterial = BaseDotMaterial;
        ClearProperties();
    }


    public void SetProperties(MaterialPropertyBlock propertyBlock)
    {
        meshRenderer.SetPropertyBlock(propertyBlock);
    }


    public void SetArrowProperties(MaterialPropertyBlock propertyBlock)
    {
        if(isArrow)
        {
            arrowMeshRenderer.SetPropertyBlock(propertyBlock);
        }
        else
        {
            dotMeshRenderer.SetPropertyBlock(propertyBlock);
        }
    }


    private void ClearProperties()
    {
        meshRenderer.SetPropertyBlock(null);
        arrowMeshRenderer.SetPropertyBlock(null);
        dotMeshRenderer.SetPropertyBlock(null);
    }


    private void Awake()
    {
        baseArrowMaterial = arrowMeshRenderer.sharedMaterial;
        baseDotMaterial = dotMeshRenderer.sharedMaterial;
    }


    public void SetArrow(bool useArrow)
    {
        isArrow = useArrow;

        arrowMeshRenderer.gameObject.SetActive(isArrow);
        dotMeshRenderer.gameObject.SetActive(!isArrow);
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

        arrowMeshRenderer.gameObject.SetActive(false);
        dotMeshRenderer.gameObject.SetActive(false);
        outlineRenderer.gameObject.SetActive(false);
        meshRenderer.enabled = false;
        Visible = false;
    }


    public void EnableVisual()
    {
        if(Visible) return;

        SetArrow(isArrow);
        SetOutline(outline);
        meshRenderer.enabled = true;
        Visible = true;
    }
}