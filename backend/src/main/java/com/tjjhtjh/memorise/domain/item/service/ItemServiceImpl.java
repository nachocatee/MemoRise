package com.tjjhtjh.memorise.domain.item.service;

import com.tjjhtjh.memorise.domain.item.repository.ItemRepository;
import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Transactional
public class ItemServiceImpl implements ItemService {

    private final ItemRepository itemRepository;

    @Override
    public Item registItem(Long itemSeq, String itemImage) {
        return itemRepository.save(new Item(itemSeq, itemImage));
    }
}
