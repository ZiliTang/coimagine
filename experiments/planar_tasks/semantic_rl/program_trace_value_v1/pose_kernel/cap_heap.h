// Frozen CAP heap ordering and last scalar trace writes.
#ifndef CAP_HEAP_H
#define CAP_HEAP_H

typedef struct { double priority; int node; } Entry;
typedef struct {
    Entry *entries;
    int size, last_push_index, last_pop_size, last_positive_pop_size;
} Heap;

static void push(Heap *h, double priority, int node) {
    int i = h->size++;
    h->last_push_index = i;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->entries[parent].priority <= priority) break;
        h->entries[i] = h->entries[parent];
        i = parent;
    }
    h->entries[i] = (Entry){priority, node};
}

static int pop(Heap *h) {
    int node = h->entries[0].node;
    Entry tail = h->entries[--h->size];
    h->last_pop_size = h->size;
    if (h->size > 0) {
        h->last_positive_pop_size = h->size;
        int i = 0;
        while (2*i + 1 < h->size) {
            int j = 2*i + 1;
            if (j+1 < h->size && h->entries[j+1].priority < h->entries[j].priority) ++j;
            if (h->entries[j].priority >= tail.priority) break;
            h->entries[i] = h->entries[j];
            i = j;
        }
        h->entries[i] = tail;
    }
    return node;
}

#endif
