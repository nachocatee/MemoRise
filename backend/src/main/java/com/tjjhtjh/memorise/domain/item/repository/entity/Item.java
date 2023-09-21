package com.tjjhtjh.memorise.domain.item.repository.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.DynamicInsert;
import org.springframework.data.annotation.CreatedDate;

import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@DynamicInsert
public class Item {

    @Id
    @Column(name = "item_seq")
    private Long itemSeq;

    private String itemImage;

    @CreatedDate
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @Builder
    public Item(Long itemSeq, String itemImage) {
        this.itemSeq = itemSeq;
        this.itemImage = itemImage;
    }
}
