using System.Collections.Generic;
using UnityEngine;

public class ObjectPool : MonoBehaviour
{
    public List<GameObject> AvailableObjects = new List<GameObject>();
    public List<GameObject> ActiveObjects = new List<GameObject>();

    private readonly Queue<GameObject> availableObjectQueue = new Queue<GameObject>();
    private readonly Dictionary<GameObject, int> availableObjectCounts = new Dictionary<GameObject, int>();
    private readonly Dictionary<GameObject, int> availableObjectIndices = new Dictionary<GameObject, int>();
    private readonly Dictionary<GameObject, int> activeObjectCounts = new Dictionary<GameObject, int>();
    private readonly Dictionary<GameObject, int> activeObjectIndices = new Dictionary<GameObject, int>();

    private int knownAvailableCount = -1;
    private int knownActiveCount = -1;

    public int PoolSize { get; private set; }

    [SerializeField] private GameObject prefab;
    [SerializeField] private int startSize;


    public void SetPoolSize(int newSize)
    {
        //This will set the target size of the pool, adding or removing objects as necessary to reach that size
        PoolSize = newSize;
        AttemptMatchPoolSize();
    }


    private void AttemptMatchPoolSize()
    {
        EnsureTrackingCurrent();

        int actualSize = AvailableObjects.Count + ActiveObjects.Count;
        int difference = actualSize - PoolSize;
        if(actualSize > PoolSize)
        {
            //Loop through AvailableObjects and delete as many as needed/able to
            int deleted = 0;
            for(int i = AvailableObjects.Count - 1; i >= 0; i--)
            {
                if(deleted >= difference)
                {
                    //Enough objects have been deleted
                    break;
                }

                GameObject deletedObject = RemoveAvailableObjectAt(i);
                Destroy(deletedObject);
                deleted++;
            }
        }
        else
        {
            //Create as many objects as needed to fill the pool
            for(int i = 0; i < Mathf.Abs(difference); i++)
            {
                GameObject newObject = CreateNewObject();
                AddAvailableObject(newObject);
            }
        }
    }


    private GameObject CreateNewObject()
    {
        //Instantiated objects are set inactive by default
        //It's the caller's responsibility to activate the object, set its parent,
        //and any other initialization that needs to happen
        GameObject newObject = Instantiate(prefab);
        newObject.transform.SetParent(transform);
        newObject.SetActive(false);

        return newObject;
    }

    private void EnsureTrackingCurrent()
    {
        if(knownAvailableCount == AvailableObjects.Count && knownActiveCount == ActiveObjects.Count)
        {
            return;
        }

        RebuildTracking();
    }


    private void RebuildTracking()
    {
        availableObjectQueue.Clear();
        availableObjectCounts.Clear();
        availableObjectIndices.Clear();
        activeObjectCounts.Clear();
        activeObjectIndices.Clear();

        for(int i = 0; i < AvailableObjects.Count; i++)
        {
            GameObject availableObject = AvailableObjects[i];
            AddTrackedObject(availableObjectCounts, availableObject);
            availableObjectIndices[availableObject] = i;
            availableObjectQueue.Enqueue(availableObject);
        }

        for(int i = 0; i < ActiveObjects.Count; i++)
        {
            GameObject activeObject = ActiveObjects[i];
            AddTrackedObject(activeObjectCounts, activeObject);
            activeObjectIndices[activeObject] = i;
        }

        RememberListCounts();
    }


    private void RememberListCounts()
    {
        knownAvailableCount = AvailableObjects.Count;
        knownActiveCount = ActiveObjects.Count;
    }


    private static void AddTrackedObject(Dictionary<GameObject, int> objectCounts, GameObject gameObject)
    {
        int count;
        objectCounts.TryGetValue(gameObject, out count);
        objectCounts[gameObject] = count + 1;
    }


    private static int RemoveTrackedObject(Dictionary<GameObject, int> objectCounts, GameObject gameObject)
    {
        int count = objectCounts[gameObject] - 1;
        if(count <= 0)
        {
            objectCounts.Remove(gameObject);
            return 0;
        }

        objectCounts[gameObject] = count;
        return count;
    }


    private void AddAvailableObject(GameObject gameObject)
    {
        availableObjectIndices[gameObject] = AvailableObjects.Count;
        AvailableObjects.Add(gameObject);
        availableObjectQueue.Enqueue(gameObject);
        AddTrackedObject(availableObjectCounts, gameObject);
        RememberListCounts();
    }


    private GameObject TakeAvailableObject()
    {
        while(availableObjectQueue.Count > 0)
        {
            GameObject gameObject = availableObjectQueue.Dequeue();
            if(availableObjectCounts.ContainsKey(gameObject))
            {
                RemoveAvailableObject(gameObject);
                return gameObject;
            }
        }

        GameObject fallbackObject = AvailableObjects[0];
        RemoveAvailableObjectAt(0);
        return fallbackObject;
    }


    private void RemoveAvailableObject(GameObject gameObject)
    {
        RemoveAvailableObjectAt(availableObjectIndices[gameObject]);
    }


    private GameObject RemoveAvailableObjectAt(int index)
    {
        GameObject gameObject = AvailableObjects[index];
        int lastIndex = AvailableObjects.Count - 1;

        if(index != lastIndex)
        {
            GameObject lastObject = AvailableObjects[lastIndex];
            AvailableObjects[index] = lastObject;
            availableObjectIndices[lastObject] = index;
        }

        AvailableObjects.RemoveAt(lastIndex);

        if(RemoveTrackedObject(availableObjectCounts, gameObject) == 0)
        {
            availableObjectIndices.Remove(gameObject);
        }
        else
        {
            availableObjectIndices[gameObject] = FindAvailableObjectIndex(gameObject);
        }

        RememberListCounts();
        return gameObject;
    }


    private int FindAvailableObjectIndex(GameObject gameObject)
    {
        for(int i = AvailableObjects.Count - 1; i >= 0; i--)
        {
            if(AvailableObjects[i] == gameObject)
            {
                return i;
            }
        }

        return -1;
    }


    private void AddActiveObject(GameObject gameObject)
    {
        activeObjectIndices[gameObject] = ActiveObjects.Count;
        ActiveObjects.Add(gameObject);
        AddTrackedObject(activeObjectCounts, gameObject);
        RememberListCounts();
    }


    private void RemoveActiveObject(GameObject gameObject)
    {
        int index = activeObjectIndices[gameObject];
        int lastIndex = ActiveObjects.Count - 1;

        if(index != lastIndex)
        {
            GameObject lastObject = ActiveObjects[lastIndex];
            ActiveObjects[index] = lastObject;
            activeObjectIndices[lastObject] = index;
        }

        ActiveObjects.RemoveAt(lastIndex);

        if(RemoveTrackedObject(activeObjectCounts, gameObject) == 0)
        {
            activeObjectIndices.Remove(gameObject);
        }
        else
        {
            activeObjectIndices[gameObject] = FindActiveObjectIndex(gameObject);
        }

        RememberListCounts();
    }


    private int FindActiveObjectIndex(GameObject gameObject)
    {
        for(int i = ActiveObjects.Count - 1; i >= 0; i--)
        {
            if(ActiveObjects[i] == gameObject)
            {
                return i;
            }
        }

        return -1;
    }


    public GameObject GetObject()
    {
        EnsureTrackingCurrent();

        if(AvailableObjects.Count > 0)
        {
            //There is an object available in the pool. Activate it and return it.
            GameObject collectedObject = TakeAvailableObject();

            AddActiveObject(collectedObject);

            return collectedObject;
        }

        //There are no available objects in the pool, so a new one will have to be created
        //This will indefinitely increase the size of the pool until it's cleared or otherwise modified
        GameObject newObject = CreateNewObject();

        AddActiveObject(newObject);
        PoolSize++;

        return newObject;
    }


    public void ReleaseObject(GameObject gameObject)
    {
        EnsureTrackingCurrent();

        if(!activeObjectCounts.ContainsKey(gameObject))
        {
            //Oops haha how did that happen
            if(!availableObjectCounts.ContainsKey(gameObject))
            {
                //Only want to destroy objects that don't exist anywhere
                Destroy(gameObject);
            }
            else
            {
                gameObject.transform.SetParent(transform);
                gameObject.SetActive(false);
                AddAvailableObject(gameObject);
            }
            return;
        }

        gameObject.transform.SetParent(transform);
        gameObject.SetActive(false);

        RemoveActiveObject(gameObject);
        AddAvailableObject(gameObject);
    }


    private void Update()
    {
        int actualSize = AvailableObjects.Count + ActiveObjects.Count;
        if(actualSize != PoolSize)
        {
            AttemptMatchPoolSize();
        }
    }


    private void Start()
    {
        SetPoolSize(startSize);
    }
}