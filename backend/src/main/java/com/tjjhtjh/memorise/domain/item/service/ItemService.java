package com.tjjhtjh.memorise.domain.item.service;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;

public interface ItemService {
    Item registItem(Long itemSeq, String itemImage);

}
