using System.Collections.Generic;
using UnityEngine;

public abstract class ObjectPool<T> : MonoBehaviour where T : MonoBehaviour
{
    public List<T> AvailableObjects = new List<T>();
    public List<T> ActiveObjects = new List<T>();

    private readonly Queue<T> availableObjectQueue = new Queue<T>();
    private readonly Dictionary<T, int> availableObjectCounts = new Dictionary<T, int>();
    private readonly Dictionary<T, int> availableObjectIndices = new Dictionary<T, int>();
    private readonly Dictionary<T, int> activeObjectCounts = new Dictionary<T, int>();
    private readonly Dictionary<T, int> activeObjectIndices = new Dictionary<T, int>();

    private int knownAvailableCount = -1;
    private int knownActiveCount = -1;

    public int PoolSize { get; private set; }

    [SerializeField] T prefab;
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

                T deletedObject = RemoveAvailableObjectAt(i);
                Destroy(deletedObject.gameObject);
                deleted++;
            }
        }
        else
        {
            //Create as many objects as needed to fill the pool
            for(int i = 0; i < Mathf.Abs(difference); i++)
            {
                T newObject = CreateNewObject();
                AddAvailableObject(newObject);
            }
        }
    }


    private T CreateNewObject()
    {
        //Instantiated objects are set inactive by default
        //It's the caller's responsibility to activate the object, set its parent,
        //and any other initialization that needs to happen
        T newItem = Instantiate(prefab);
        newItem.transform.SetParent(transform);
        newItem.gameObject.SetActive(false);

        return newItem;
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
            T availableObject = AvailableObjects[i];
            AddTrackedObject(availableObjectCounts, availableObject);
            availableObjectIndices[availableObject] = i;
            availableObjectQueue.Enqueue(availableObject);
        }

        for(int i = 0; i < ActiveObjects.Count; i++)
        {
            T activeObject = ActiveObjects[i];
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


    private static void AddTrackedObject(Dictionary<T, int> objectCounts, T target)
    {
        int count;
        objectCounts.TryGetValue(target, out count);
        objectCounts[target] = count + 1;
    }


    private static int RemoveTrackedObject(Dictionary<T, int> objectCounts, T target)
    {
        int count = objectCounts[target] - 1;
        if(count <= 0)
        {
            objectCounts.Remove(target);
            return 0;
        }

        objectCounts[target] = count;
        return count;
    }


    private void AddAvailableObject(T target)
    {
        availableObjectIndices[target] = AvailableObjects.Count;
        AvailableObjects.Add(target);
        availableObjectQueue.Enqueue(target);
        AddTrackedObject(availableObjectCounts, target);
        RememberListCounts();
    }


    private T TakeAvailableObject()
    {
        while(availableObjectQueue.Count > 0)
        {
            T target = availableObjectQueue.Dequeue();
            if(availableObjectCounts.ContainsKey(target))
            {
                RemoveAvailableObject(target);
                return target;
            }
        }

        T fallbackObject = AvailableObjects[0];
        RemoveAvailableObjectAt(0);
        return fallbackObject;
    }


    private void RemoveAvailableObject(T target)
    {
        RemoveAvailableObjectAt(availableObjectIndices[target]);
    }


    private T RemoveAvailableObjectAt(int index)
    {
        T target = AvailableObjects[index];
        int lastIndex = AvailableObjects.Count - 1;

        if(index != lastIndex)
        {
            T lastObject = AvailableObjects[lastIndex];
            AvailableObjects[index] = lastObject;
            availableObjectIndices[lastObject] = index;
        }

        AvailableObjects.RemoveAt(lastIndex);

        if(RemoveTrackedObject(availableObjectCounts, target) == 0)
        {
            availableObjectIndices.Remove(target);
        }
        else
        {
            availableObjectIndices[target] = FindAvailableObjectIndex(target);
        }

        RememberListCounts();
        return target;
    }


    private int FindAvailableObjectIndex(T target)
    {
        for(int i = AvailableObjects.Count - 1; i >= 0; i--)
        {
            if(EqualityComparer<T>.Default.Equals(AvailableObjects[i], target))
            {
                return i;
            }
        }

        return -1;
    }


    private void AddActiveObject(T target)
    {
        activeObjectIndices[target] = ActiveObjects.Count;
        ActiveObjects.Add(target);
        AddTrackedObject(activeObjectCounts, target);
        RememberListCounts();
    }


    private void RemoveActiveObject(T target)
    {
        int index = activeObjectIndices[target];
        int lastIndex = ActiveObjects.Count - 1;

        if(index != lastIndex)
        {
            T lastObject = ActiveObjects[lastIndex];
            ActiveObjects[index] = lastObject;
            activeObjectIndices[lastObject] = index;
        }

        ActiveObjects.RemoveAt(lastIndex);

        if(RemoveTrackedObject(activeObjectCounts, target) == 0)
        {
            activeObjectIndices.Remove(target);
        }
        else
        {
            activeObjectIndices[target] = FindActiveObjectIndex(target);
        }

        RememberListCounts();
    }


    private int FindActiveObjectIndex(T target)
    {
        for(int i = ActiveObjects.Count - 1; i >= 0; i--)
        {
            if(EqualityComparer<T>.Default.Equals(ActiveObjects[i], target))
            {
                return i;
            }
        }

        return -1;
    }


    public T GetObject()
    {
        EnsureTrackingCurrent();

        if(AvailableObjects.Count > 0)
        {
            //There is an object available in the pool. Activate it and return it.
            T collectedObject = TakeAvailableObject();

            AddActiveObject(collectedObject);

            return collectedObject;
        }

        //There are no available objects in the pool, so a new one will have to be created
        //This will indefinitely increase the size of the pool until it's cleared or otherwise modified
        T newObject = CreateNewObject();

        AddActiveObject(newObject);
        PoolSize++;

        return newObject;
    }


    public void ReleaseObject(T target)
    {
        EnsureTrackingCurrent();

        if(!activeObjectCounts.ContainsKey(target))
        {
            //Oops haha how did that happen
            if(!availableObjectCounts.ContainsKey(target))
            {
                //Only want to destroy objects that don't exist anywhere
                Destroy(target.gameObject);
            }
            else
            {
                target.gameObject.transform.SetParent(transform);
                target.gameObject.SetActive(false);
            }
            return;
        }

        target.gameObject.transform.SetParent(transform);
        target.gameObject.SetActive(false);

        RemoveActiveObject(target);
        AddAvailableObject(target);
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