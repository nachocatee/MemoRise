package com.tjjhtjh.memorise.domain.item.service;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;

public interface ItemService {
    Item registItem(String itemName, String itemImage);
}
