using UnityEngine;

public class BombHandler : MonoBehaviour
{
    public bool Visible => meshRenderer.enabled;

    [SerializeField] public AudioSource audioSource;
    [SerializeField] private MeshRenderer meshRenderer;
    [SerializeField] private MeshRenderer outlineRenderer;

    private bool outline;
    private Color outlineColor;

    private MaterialPropertyBlock outlineProperties;


    public void SetMaterial(Material newMaterial)
    {
        meshRenderer.sharedMaterial = newMaterial;
        ClearProperties();
    }


    public void SetProperties(MaterialPropertyBlock propertyBlock)
    {
        meshRenderer.SetPropertyBlock(propertyBlock);
    }


    public void ClearProperties()
    {
        meshRenderer.SetPropertyBlock(null);
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
        outlineRenderer.gameObject.SetActive(false);
        meshRenderer.enabled = false;
    }


    public void EnableVisual()
    {
        SetOutline(outline);
        meshRenderer.enabled = true;
    }
}