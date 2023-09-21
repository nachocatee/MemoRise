package com.tjjhtjh.memorise.domain.item.repository.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;

import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Item {

    @Id
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
